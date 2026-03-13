# M-Guide | Maternal Diabetes Prevention Platform

This is a Streamlit demo that combines:

1. **ANC booking risk of GDM** from a saved CatBoost model and saved scaler
2. **Pregnancy after GDM** future T2DM risk from the antenatal published logistic equation
3. **Postnatal follow-up** future T2DM risk from the postnatal published logistic equation

## Included public-demo features

- Monash-style branding
- custom logo area
- footer with institution details
- QR code cards for public app, GitHub repo, and publication
- PDF-style report page
- GitHub / Streamlit Cloud deployment-ready files

## Files you need in the project root

- `app.py`
- `requirements.txt`
- `All_data_Catboost_GDM_ML_model.bin`
- `All_data_scaler.pkl`

Optional:
- `assets/custom_logo.png` or `assets/custom_logo.svg`

If no custom logo file is present, the app uses `assets/logo_placeholder.svg`.

## Run locally

```bash
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

## Streamlit Community Cloud deployment

1. Create a GitHub repository.
2. Upload all project files, including the model and scaler if this is a demo deployment.
3. In Streamlit Community Cloud, create a new app from the repo.
4. Set the main file path to `app.py`.
5. After deployment, paste the public app URL into the app sidebar so the QR code and report page use the live URL.

## Branding

To replace the placeholder logo:

- add `assets/custom_logo.png`, or
- add `assets/custom_logo.svg`

## Public release reminder

This project is a **research demo**. Before sharing outside a controlled prototype setting, confirm:

- approved institutional branding
- governance and security review
- validation and clinical approval
- correct public wording and disclaimer
