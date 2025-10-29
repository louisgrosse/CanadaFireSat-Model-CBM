from src.CBM.concepts_minimal import name_sae_concepts

sae_ckpt = "/home/louis/Code/CanadaFireSat-Model-CBM/results/topk_overcomplete_runs/SAE_training_OA/2025-10-27_11-57-14/epoch_epoch=019.ckpt"        # trained with train_sae.py / plSAE
concept_csv_path = "/home/louis/Code/wildfire-forecast/sae_features_mm/"
concept_csv = concept_csv_path+ "wildfire_cbm_concepts_generic_5k.csv"       # has column: concept, category


df = name_sae_concepts(
    sae_ckpt=sae_ckpt,       # config.yaml must be alongside
    csv_path=concept_csv,
    text_col="concept",
    device="cuda",
    out_csv=concept_csv_path+ "/concept_names.csv",
)
