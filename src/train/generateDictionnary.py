from concepts_minimal import name_sae_concepts

sae_ckpt = "path/to/your_sae.ckpt"        # trained with train_sae.py / plSAE
concept_csv = "/work/eceo/grosse/saeFeatures/wildfire_cbm_concepts_generic_5k.csv"       # has column: concept, category

df = name_sae_concepts(
    sae_ckpt=sae_ckpt,
    csv_path=concept_csv,
    text_col="concept",          
    topk=5,                      
    model_name="Llama3-MS-CLIP-Base",  
    msclip_ckpt=None,            
    device="cuda",              
    out_csv="concept_names.csv"  
)
print(df.head())
