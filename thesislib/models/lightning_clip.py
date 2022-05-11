import torch
import torchmetrics
from clip import clip
from pytorch_lightning import LightningModule
from torch.nn.functional import cross_entropy


class LightningClip(LightningModule):

    def __init__(self, clip_model, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.image_encoder = clip_model.visual
        self.text_encoder = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.logit_scale = clip_model.logit_scale
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.token_embedding = clip_model.token_embedding
        self.image_accuracy = torchmetrics.Accuracy()
        self.text_accuracy = torchmetrics.Accuracy()

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_encoder(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def encode_image(self, image):
        return self.image_encoder(image.type(self.dtype))

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

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
        return torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)

    def _clip_step(self, batch):
        images, captions, *other = batch
        tokenized_captions = clip.tokenize(captions).to(self.device)
        logits_per_image, logits_per_text = self(images, tokenized_captions)
        return logits_per_image, logits_per_text

    def training_step(self, batch, batch_idx):
        logits_per_image, logits_per_text = self._clip_step(batch)
        labels = torch.arange(len(logits_per_image)).to(self.device)
        image_loss = cross_entropy(logits_per_image, labels)
        text_loss = cross_entropy(logits_per_text, labels)
        loss = (image_loss + text_loss) / 2
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_index):
        logits_per_image, logits_per_text = self._clip_step(batch)
        labels = torch.arange(len(logits_per_image)).to(self.device)
        image_loss = cross_entropy(logits_per_image, labels)
        text_loss = cross_entropy(logits_per_text, labels)
        loss = (image_loss + text_loss) / 2
        self.image_accuracy(logits_per_image, labels)
        self.text_accuracy(logits_per_text, labels)
        self.log("val_loss", loss)
        self.log("val_image_accuracy", self.image_accuracy, on_step=False, on_epoch=True)
        self.log("val_text_accuracy", self.text_accuracy, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        logits_per_image, logits_per_text = self._clip_step(batch)
        labels = torch.arange(len(logits_per_image)).to(self.device)
        image_loss = cross_entropy(logits_per_image, labels)
        text_loss = cross_entropy(logits_per_text, labels)
        loss = (image_loss + text_loss) / 2
        self.image_accuracy(logits_per_image, labels)
        self.text_accuracy(logits_per_text, labels)
        self.log("test_image_accuracy", self.image_accuracy, on_step=False, on_epoch=True)
        self.log("test_text_accuracy", self.text_accuracy, on_step=False, on_epoch=True)



