import clip
import torch
from pytorch_lightning import LightningModule
from torch.nn.functional import cross_entropy

from . import ContextAddition, LightningClip
from ..metrics import PartitionRecall


class DynamicClip(LightningModule):

    def __init__(self, clip_model,
                 context_length,
                 insertion,
                 scheduler,
                 validation_partition,
                 test_partition):

        super().__init__()
        self.save_hyperparameters("context_length", "insertion", "scheduler")
        self.image_encoder = clip_model.visual
        self.text_encoder = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.logit_scale = clip_model.logit_scale
        self.ln_final = clip_model.ln_final
        self.scheduler = scheduler
        self.text_projection = clip_model.text_projection

        self.context_length = context_length
        self.context_addition = ContextAddition(clip_model, context_length, insertion)
        self._freeze_components()

        self.validation_recall = PartitionRecall(validation_partition)
        self.test_recall = PartitionRecall(test_partition)

    def on_fit_start(self) -> None:
        self.image_encoder.eval()
        self.text_encoder.eval()

    def on_validation_start(self) -> None:
        self.image_encoder.eval()
        self.text_encoder.eval()

    def on_test_start(self) -> None:
        self.image_encoder.eval()
        self.text_encoder.eval()

    def _freeze_components(self):
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.context_addition.token_embedding.parameters():
            param.requires_grad = False
        for param in self.ln_final.parameters():
            param.requires_grad = False
        self.logit_scale.requires_grad = False
        self.text_projection.requires_grad = False
        self.positional_embedding.requires_grad = False

    def encode_text(self, tokenized_text, dynamic_bools):
        x = self.context_addition(tokenized_text, dynamic_bools)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_encoder(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = torch.where(dynamic_bools.unsqueeze(1),
                        x[torch.arange(x.shape[0]), tokenized_text.argmax(dim=-1) + self.context_length],
                        x[torch.arange(x.shape[0]), tokenized_text.argmax(dim=-1)]
                        )

        x = x @ self.text_projection

        return x

    def encode_image(self, image):
        return self.image_encoder(image.type(self.dtype))

    def forward(self, images, tokenized_captions, dynamic_bools):
        image_features = self.encode_image(images)
        text_features = self.encode_text(tokenized_captions, dynamic_bools)
        assert torch.any(dynamic_bools)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=1e-3)
        if self.scheduler:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[120, 160, 200])
            return [optimizer], [scheduler]
        return optimizer

    def training_step(self, batch, batch_idx):
        loss = self._clip_step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, indices = batch
        dataset = self.trainer.val_dataloaders[0].dataset
        tokenized_captions = dataset.tokenized_captions
        tokenized_captions = [tokenized_split.to(self.device) for tokenized_split in tokenized_captions]
        text_bools = [split.to(self.device) for split in dataset.split_bools]
        logits = []
        for i, tokenized_split in enumerate(tokenized_captions):
            logits_per_image, logits_per_text = self(images, tokenized_split, text_bools[i])
            logits.append(logits_per_image)
        combined_logits = torch.cat(logits, dim=1)
        self.validation_recall.update(combined_logits)

    def validation_epoch_end(self, outputs):
        dataset = self.trainer.val_dataloaders[0].dataset
        self.validation_recall.txt2vis = dataset.txt2vis
        eval_dict = self.validation_recall.compute()
        self.log('validation_dynamic_r@1', eval_dict["dynamic"][0])
        self.log('validation_static_r@1', eval_dict["static"][0])
        self.log('validation_total_r@1', eval_dict["total"][0], prog_bar=True)
        self.log('validation_dynamic_r@5', eval_dict["dynamic"][1])
        self.log('validation_static_r@5', eval_dict["static"][1])
        self.log('validation_total_r@5', eval_dict["total"][1])
        self.validation_recall.reset()

    def _clip_step(self, batch):
        images, captions, dynamic_bools = batch
        tokenized_captions = clip.tokenize(captions).to(self.device)
        logits_per_image, logits_per_text = self(images, tokenized_captions, dynamic_bools)
        labels = torch.arange(len(logits_per_image)).to(self.device)
        image_loss = cross_entropy(logits_per_image, labels)
        text_loss = cross_entropy(logits_per_text, labels)
        loss = (image_loss + text_loss) / 2
        return loss

    def test_step(self, batch, batch_idx):
        images, indices = batch
        dataset = self.trainer.test_dataloaders[0].dataset
        tokenized_captions = dataset.tokenized_captions
        tokenized_captions = [tokenized_split.to(self.device) for tokenized_split in tokenized_captions]
        text_bools = [split.to(self.device) for split in dataset.split_bools]
        logits = []
        for i, tokenized_split in enumerate(tokenized_captions):
            logits_per_image, logits_per_text = self(images, tokenized_split, text_bools[i])
            logits.append(logits_per_image)
        combined_logits = torch.cat(logits, dim=1)
        self.test_recall.update(combined_logits)

    def test_epoch_end(self, outputs):
        dataset = self.trainer.test_dataloaders[0].dataset
        self.test_recall.txt2vis = dataset.txt2vis
        eval_dict = self.test_recall.compute()
        self.log('test_dynamic_r@1', eval_dict["dynamic"][0])
        self.log('test_static_r@1', eval_dict["static"][0])
        self.log('test_total_r@1', eval_dict["total"][0])
        self.log('test_dynamic_r@5', eval_dict["dynamic"][1])
        self.log('test_static_r@5', eval_dict["static"][1])
        self.log('test_total_r@5', eval_dict["total"][1])
        self.test_recall.reset()


def transfer_clip_modules(lightning_clip: LightningClip, dynamic_clip: DynamicClip):
    dynamic_clip.text_encoder = lightning_clip.text_encoder
    dynamic_clip.image_encoder = lightning_clip.image_encoder
    dynamic_clip.positional_embedding = lightning_clip.positional_embedding
    dynamic_clip.logit_scale = lightning_clip.logit_scale
    dynamic_clip.ln_final = lightning_clip.ln_final
    dynamic_clip.text_projection = lightning_clip.text_projection

