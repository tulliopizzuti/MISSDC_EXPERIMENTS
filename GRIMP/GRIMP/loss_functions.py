import torch
import torch.nn.functional as F


def compute_loss(pos_score, neg_score, device="cpu"):
    # if neg_score.shape[0] > pos_score.shape[0]:
    #     factor = neg_score.shape[0]//pos_score.shape[0]
    #     pos_score = pos_score.repeat(factor)

    scores = torch.cat([pos_score, neg_score]).to(device)
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    ).to(device)
    # return F.binary_cross_entropy_with_logits(scores, labels)
    # Using bce_with logits automatically takes sigmoid over scores.
    # Now using bce directly so that could be done manually
    return F.binary_cross_entropy(scores, labels)


def compute_loss_max_score(pos_score, neg_score, device="cpu"):
    # pos_res = torch.argmax(pos_score, dim=1)
    # neg_res = torch.argmax(neg_score, dim=1)
    # scores = torch.cat([pos_res, neg_res]).to(float).requires_grad_(True)
    #

    # l1 = torch.cat([torch.ones(pos_score.shape[0],1), torch.zeros(neg_score.shape[0],1)]).to(device)
    # l2 = torch.cat([torch.zeros(pos_score.shape[0],1), torch.ones(neg_score.shape[0],1)]).to(device)
    # labels = torch.cat([l1, l2], dim=1)
    scores = torch.cat([pos_score, neg_score]).requires_grad_(True)

    labels = (
        torch.cat([torch.zeros(pos_score.shape[0]), torch.ones(neg_score.shape[0])])
        .to(int)
        .to(device)
    )
    #
    return F.cross_entropy(scores, labels)


def compute_loss_binary_cross_entropy_logits(pos_score, neg_score, device="cpu"):
    scores = torch.cat([pos_score, neg_score])
    labels = (
        torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
        .to(int)
        .to(device)
    )
    return F.binary_cross_entropy_with_logits(scores, labels)


def compute_hinge_loss(pos_score, neg_score, margin=1, device="cpu"):
    if neg_score.shape[0] > pos_score.shape[0]:
        factor = neg_score.shape[0] // pos_score.shape[0]

        pos_score = pos_score.repeat(factor)
    scores = torch.cat([pos_score, neg_score]).to(device)
    labels = (
        torch.cat([-torch.ones(pos_score.shape[0]), torch.ones(neg_score.shape[0])])
        .to(int)
        .to(device)
    )

    # loss = pos_score - neg_score
    loss = F.hinge_embedding_loss(scores, labels, margin=margin, reduction="mean")

    return loss


def compute_loss_cosine(pos_score, neg_score, graph_dataset, h):
    pos_labels = h[graph_dataset.labels].squeeze()
    neg_labels = pos_labels.repeat(
        graph_dataset.positive_negative_train_scale_factor, 1
    )
    labels = torch.cat([pos_labels, neg_labels])
    scores = torch.cat([pos_score, neg_score])
    flags = torch.cat(
        [torch.ones(pos_labels.shape[0]), -torch.ones(neg_labels.shape[0])]
    )
    loss = F.cosine_embedding_loss(scores, labels, flags)
    # loss = F.cosine_embedding_loss(pos_score, pos_labels, torch.ones(pos_labels.shape[0]))
    return loss

    all_loss = 1 - F.cosine_similarity(pos_score, pos_labels)
    return all_loss.mean()


def compute_loss_dot(pos_score, graph_dataset, h):
    pos_labels = h[graph_dataset.labels].squeeze()
    dot = torch.dot(pos_score, pos_labels)
    return 1 - dot


def compute_loss_multilabel(pos_score, labels, device="cpu"):
    # score = torch.argmax(pos_score, dim=1).to(int).unsqueeze(1).to(device)
    res = F.cross_entropy(pos_score, labels, reduction="mean").to(device)
    return res


def compute_loss_mse(pos_score, labels, target_column, device="cpu"):
    res = F.mse_loss(pos_score.squeeze(1), labels[target_column].to(device))
    return res


def compute_loss_multilabel_attr(pos_score, labels, target_column, device="cpu"):
    # score = torch.argmax(pos_score, dim=1).to(int).unsqueeze(1).to(device)
    res = F.cross_entropy(pos_score.to(device), labels[target_column].to(device))
    return res


def compute_focal_loss(
    pos_score: torch.Tensor,
    graph_dataset,
    stage,
    target_column=None,
    alpha: float = 0.7,
    gamma: float = 2,
    reduction: str = "mean",
    device="cpu",
):
    if stage == "train":
        labels = graph_dataset.labels
        p_t = graph_dataset.weights
    elif stage == "valid":
        labels = graph_dataset.labels_valid
        p_t = graph_dataset.weights_valid
    else:
        raise ValueError(f"Unknown stage {stage}")

    # basic multilabel (not selecting by column)
    if target_column is None:
        ce_loss = F.cross_entropy(
            pos_score.to(device), labels.to(device), reduction="none"
        ).unsqueeze(1)
        p_t = p_t.to(device)
        loss = ce_loss * ((1 - p_t) ** gamma)
        loss = loss.to(device)
        if alpha >= 0:
            alpha_t = alpha * p_t + (1 - alpha) * (1 - p_t)
            loss = alpha_t * loss
    else:
        ce_loss = F.cross_entropy(
            pos_score.to(device), labels[target_column].to(device), reduction="none"
        ).unsqueeze(1)
        p_t[target_column] = p_t[target_column].to(device)
        if 1 >= alpha >= 0:
            loss = ce_loss * (alpha * (1 - p_t[target_column]) ** gamma)
            loss = loss.to(device)
            # alpha_t = alpha *
            #
            # # alpha_t = alpha * p_t[target_column] + (1 - alpha) * (1 - p_t[target_column])
            # loss = alpha_t * loss
        else:
            raise ValueError(f"Invalid alpha: {alpha}")

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss
