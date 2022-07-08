import torch


def insert_tca_vectors(
        context_vectors,
        embeddings,
        eot_indices,
        mode
):
    if mode == 'prefix':
        return insert_prefix_vectors(context_vectors, embeddings)
    elif mode == 'postfix':
        return insert_postfix_vectors(context_vectors,
                                      embeddings,
                                      eot_indices)
    elif mode == 'infix':
        return insert_infix_vectors(context_vectors,
                                    embeddings,
                                    eot_indices)


def insert_prefix_vectors(context_vectors, embeddings):
    corrected_embeddings = torch.cat(
        [embeddings[:, :1, :],
         context_vectors,
         embeddings[:, 1:(77 - len(context_vectors)), :]],
        dim=1
    )

    return corrected_embeddings


def insert_postfix_vectors(context_vectors, embeddings, eot_indices):
    batch_size = embeddings.shape[0]
    corrected_embeddings = torch.zeros_like(embeddings).type_as(embeddings)
    for idx in range(batch_size):
        eot_index = eot_indices[idx].item()
        corrected_embeddings[idx, :, :] = torch.cat(
            [embeddings[idx, :eot_index, :],
             context_vectors[idx],
             embeddings[idx, eot_index:(77 - context_vectors.shape[1]), :]]
        )

    return corrected_embeddings


def insert_infix_vectors(context_vectors, embeddings, eot_indices):
    batch_size = embeddings.shape[0]
    partial_length = context_vectors.shape[1] // 2
    corrected_embeddings = torch.zeros_like(embeddings).type_as(embeddings)
    for idx in range(batch_size):
        eot_index = eot_indices[idx].item()
        corrected_embeddings[idx, :, :] = torch.cat(
            [embeddings[idx, :1, :],
             context_vectors[idx, :partial_length, :],
             embeddings[idx, 1:eot_index, :],
             context_vectors[idx, partial_length:, :],
             embeddings[idx, eot_index:(77 - context_vectors.shape[1]), :]]
        )

    return corrected_embeddings
