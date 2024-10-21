import os
import fsspec
import torch
import torch.nn as nn
import clip
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

class TransformerWithToken(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers):
        super().__init__()

        self.token = nn.Parameter(torch.randn(1, 1, d_model))
        token_mask = torch.zeros(1, 1, dtype=torch.bool)
        self.register_buffer("token_mask", token_mask)

        self.core = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
            ),
            num_layers=num_layers,
        )

    def forward(self, x, src_key_padding_mask):
        # x: [N, B, E]
        # padding_mask: [B, N]
        #   `False` for valid values
        #   `True` for padded values

        B = x.size(1)

        token = self.token.expand(-1, B, -1)
        x = torch.cat([token, x], dim=0)

        token_mask = self.token_mask.expand(B, -1)
        padding_mask = torch.cat([token_mask, src_key_padding_mask], dim=1)

        x = self.core(x, src_key_padding_mask=padding_mask)

        return x


class FIDNetV3(nn.Module):
    def __init__(self, d_model=512, nhead=4, num_layers=4, max_bbox=50):
        super().__init__()

        # encoder
        # self.emb_label = nn.Embedding(num_label, d_model)
        clip_model, preprocess = clip.load('ViT-B/32')
        self.emb_label = clip_model
        self.emb_label.eval()

        self.fc_bbox = nn.Linear(4, d_model)
        self.enc_fc_in = nn.Linear(d_model * 2, d_model)

        self.enc_transformer = TransformerWithToken(
            d_model=d_model,
            dim_feedforward=d_model // 2,
            nhead=nhead,
            num_layers=num_layers,
        )

        self.fc_out_disc = nn.Linear(d_model, 1)

        # decoder
        self.pos_token = nn.Parameter(torch.rand(max_bbox, 1, d_model))
        self.dec_fc_in = nn.Linear(d_model * 2, d_model)

        te = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model // 2
        )
        self.dec_transformer = nn.TransformerEncoder(te, num_layers=num_layers)

        self.fc_out_clip_feat = nn.Linear(d_model, d_model)
        self.fc_out_bbox = nn.Linear(d_model, 4)

    def extract_features(self, bbox, label, padding_mask):
        device = bbox.device
        b = self.fc_bbox(bbox)
        with torch.no_grad():
            prompts = torch.stack([clip.tokenize(l, truncate=True) for l in label]).to(device)
            text_features = torch.stack([self.emb_label.encode_text(p) for p in prompts])
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.detach().to(device)

        x = self.enc_fc_in(torch.cat([b, text_features], dim=-1))
        x = torch.relu(x).permute(1, 0, 2)
        x = self.enc_transformer(x, padding_mask)
        return x[0], text_features

    def forward(self, bbox, label, padding_mask):
        B, N, _ = bbox.size()
        x, text_features = self.extract_features(bbox, label, padding_mask)

        logit_disc = self.fc_out_disc(x).squeeze(-1)

        x = x.unsqueeze(0).expand(N, -1, -1)
        t = self.pos_token[:N].expand(-1, B, -1)
        x = torch.cat([x, t], dim=-1)
        x = torch.relu(self.dec_fc_in(x))

        x = self.dec_transformer(x, src_key_padding_mask=padding_mask)
        # x = x.permute(1, 0, 2)[~padding_mask]
        x = x.permute(1, 0, 2)

        clip_feat_pred = self.fc_out_clip_feat(x)
        clip_feat_pred = nn.functional.normalize(clip_feat_pred, dim=2, p=2)

        bbox_pred = torch.sigmoid(self.fc_out_bbox(x))

        return logit_disc, clip_feat_pred, bbox_pred, text_features


def load_fidnet_v3(dataset, weight_dir: str, device: torch.device) -> nn.Module:
    prefix = f"{dataset.name}-max{dataset.max_seq_length}"
    ckpt_path = os.path.join(weight_dir, prefix, "model_best.pth.tar")
    fid_model = FIDNetV3(max_bbox=dataset.max_seq_length).to(device)

    with fsspec.open(ckpt_path, "rb") as file_obj:
        x = torch.load(file_obj, map_location=device)
    fid_model.load_state_dict(x["state_dict"])
    fid_model.eval()
    return fid_model


if __name__=="__main__":

    l = [['car', 'train', 'apple'], ['cat', 'dog', '']]
    bbox = torch.rand([2, 3, 4])
    padding_mask = torch.tensor([[False, False, False], [False, False, True]])
    M = FIDNetV3()

    logit_disc, clip_feat_pred, bbox_pred, clip_feat = M(bbox, l, padding_mask)
    print('logit_disc', logit_disc.shape)
    print('clip_feat_pred', clip_feat_pred.shape)
    print('bbox_pred', clip_feat_pred.shape)
