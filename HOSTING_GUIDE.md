# Hosting Guide for Early Disease Predicting App

## 🚀 Quick Deploy Options

### 1. **Streamlit Cloud** (Easiest - Free)
1. Go to https://share.streamlit.io
2. Connect your GitHub account
3. Select your repository: `ashik230509/early-disease-predicting`
4. Set main file path: `app.py`
5. Click Deploy!

### 2. **Heroku** (Free tier available)
1. Install Heroku CLI: https://devcenter.heroku.com/articles/heroku-cli
2. Login: `heroku login`
3. Create app: `heroku create your-app-name`
4. Deploy: `git push heroku main`

### 3. **Railway** (Simple & Free)
1. Go to https://railway.app
2. Connect GitHub repository
3. Add environment variable: `PORT=8000`
4. Deploy automatically

### 4. **Render** (Free tier)
1. Go to https://render.com
2. Connect GitHub repo
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `streamlit run app.py --server.port $PORT --server.headless true`

## 📋 Pre-deployment Checklist

- [ ] Ensure `outputs/` folder with models exists
- [ ] Test locally: `streamlit run app.py`
- [ ] Update requirements.txt with exact versions
- [ ] Add runtime.txt for Python version (Python-3.9.13)

## 🔧 Configuration for Deployment

Add this to your `app.py` for better deployment:

```python
# Add at the top of app.py
import os

# For deployment platforms
if 'DYNO' in os.environ:  # Heroku
    port = int(os.environ.get('PORT', 8501))
elif 'RAILWAY_ENVIRONMENT' in os.environ:  # Railway
    port = int(os.environ.get('PORT', 8501))
else:
    port = 8501

# Run with: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

## 🌐 Live Examples
- **Streamlit Cloud**: https://share.streamlit.io/yourusername/yourrepo/main/app.py
- **Heroku**: https://your-app-name.herokuapp.com
- **Railway**: https://your-project.railway.app
- **Render**: https://your-service.onrender.com

Choose **Streamlit Cloud** for the easiest deployment! 🎯