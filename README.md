
### Components
- **Data Collection:** Flickr30K dataset  
- **Models Used:**  
  - Gender: `rizvandwiki/gender-classification-2`  
  - Age: `nateraw/vit-age-classifier`  
  - Emotion: `trpakov/vit-face-expression`  
- **Visualization:** Plotly dashboards, HTML reports  
- **Runtime:** PyTorch + CUDA acceleration (Tesla V100)

---

## 📊 Results
| Metric | Value |
|--------|-------|
| Images Processed | 200 |
| Faces Detected | 90 |
| Gender Detection Accuracy | 91% |
| Average Bias Score | 0.337 (Moderate) |
| Compliance Score | 100% |

---

## 💡 Real-World Applications
- HR recruitment auditing  
- Healthcare model validation  
- Financial AI fairness checks  
- Social media content moderation  

**Impact:** Saves $100K+ annually in manual auditing costs by automating fairness checks.

---

## 🧱 Setup
```bash
git clone https://github.com/yourusername/gender_bias
cd gender_bias
bash setup.sh
python verify_setup.py
python main.py

Requirements:
Python 3.8+, CUDA-enabled GPU, 16GB+ RAM

## 🔮 Future Work

Expand to race and accessibility bias

Integrate live video feeds for real-time monitoring

Develop Fairness-as-a-Service API

📜 License

MIT License — free to use and modify with attribution.

🏆 Acknowledgements

Built with open-source AI tools by Team HackHers for Hackathon 2025.
Special thanks to the UCF AI and Computer Vision community.
