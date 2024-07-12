import hydra
import os
import torch
from models.transformer import Transformer
from models.selsolcom import SelSolCom


def build_selsolcom(cfg_model, device, vocabularies):
	selector = build_transformer(cfg_model.selector, device, vocabularies['selector'])
	solver = build_transformer(cfg_model.solver, device, vocabularies['solver'])
	return SelSolCom(selector, solver, vocabularies['selsolcom'], cfg_model.n_multi, cfg_model.zoom_selector, cfg_model.length_threshold).to(device)

def build_transformer(cfg_model, device, vocabulary):
	if cfg_model.ckpt is not None:
		print(f"Loading model from ckpt: {cfg_model.ckpt}...")
		torch_ckpt = torch.load(os.path.join(hydra.utils.get_original_cwd(), f'../checkpoints/{cfg_model.ckpt}'))
		print(f"Model last saved at iteration n. {torch_ckpt['update']}.")
		if 'model_cfg' in torch_ckpt:
			print("Found model cfg in ckpt, building model using that.")
			torch_ckpt['model_cfg'].deterministic = cfg_model.deterministic
			cfg = torch_ckpt['model_cfg']
			cfg = cast_numbers(cfg)
		else:
			print("Model cfg not found in cfg, building model using input cfg.")
			cfg = cfg_model
		model = build_transformer_base(cfg, device, vocabulary)
		model.load_model_weights(cfg_model.ckpt)
		return model
	else:
		return build_transformer_base(cfg_model, device, vocabulary)
	
def build_transformer_base(cfg_model, device, vocabulary):
	if not 'mha_init_gain' in cfg_model:  # backwards compatibility
		import omegaconf

		dict_cfg = dict(cfg_model)
		dict_cfg['mha_init_gain'] = 1
		cfg_model = omegaconf.OmegaConf.create(dict_cfg)

	return Transformer(
				d_model=cfg_model.d_model,
				ff_mul=cfg_model.ff_mul,
				num_heads=cfg_model.num_heads,
				num_layers_enc=cfg_model.num_layers_enc,
				num_layers_dec=cfg_model.num_layers_dec,
				vocabulary=vocabulary,
				label_pe_enc=cfg_model.label_pe_enc,
				label_pe_dec=cfg_model.label_pe_dec,
				deterministic=cfg_model.deterministic,
				n_multi=cfg_model.n_multi,
				temperature=cfg_model.temperature,
				max_range_pe=cfg_model.max_range_pe,
				diag_mask_width_below=cfg_model.diag_mask_width_below,
				diag_mask_width_above=cfg_model.diag_mask_width_above,
				average_attn_weights=cfg_model.average_attn_weights,
				store_attn_weights=cfg_model.store_attn_weights,
				mha_init_gain=cfg_model.mha_init_gain,
				num_recurrent_steps=cfg_model.num_recurrent_steps,
				multi_fwd_threshold=cfg_model.multi_fwd_threshold,
				dropout=cfg_model.dropout,
				device=device,
			).to(device)

def cast_numbers(cfg_model):
	int_fields = ['d_model', 'ff_mul', 'num_heads', 'num_layers_enc', 'num_layers_dec', 'diag_mask_width_above']
	float_fields = ['dropout', 'mha_init_gain']

	for field in int_fields:
		if cfg_model[field] is not None:
			cfg_model[field] = int(cfg_model[field])

	for field in float_fields:
		if cfg_model[field] is not None:
			cfg_model[field] = float(cfg_model[field])

	return cfg_model