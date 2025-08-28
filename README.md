# Elephant Re-ID (Final Year Project)

This project explores the use of computer vision and deep learning to **identify and track elephants across multiple videos**.  
The main objective is to ensure that if the **same elephant** appears in different videos, it is consistently assigned the **same identification number**.

---

## 🌍 Motivation
Elephants are social animals, and their monitoring is essential for:
- **Wildlife conservation** – tracking population and migration patterns.
- **Research** – studying group behavior and interactions.
- **Anti-poaching efforts** – identifying and protecting specific individuals.

Manual tracking of elephants from large amounts of video data is time-consuming and prone to error.  
This project aims to **automate the process** by combining modern AI tools for detection, tracking, and re-identification.

---

## 🎯 Goals
1. **Detect elephants** in video footage using state-of-the-art object detection models.  
2. **Track individuals** across video frames to group images of the same elephant.  
3. **Extract unique features** (embeddings) from each elephant’s appearance.  
4. **Compare features across videos** to check if an elephant has been seen before.  
5. **Assign IDs** so that the same elephant gets the same ID consistently, even across different videos.

---

## 🧠 Approach (High-Level)
- Use a pre-trained object detection model (e.g., YOLO) to locate elephants in video frames.  
- Apply a tracking algorithm (e.g., DeepSORT) to maintain identities within a single video.  
- Use a feature extraction model (e.g., ResNet) to generate embeddings for each elephant.  
- Match embeddings across videos to recognize the same individual.  
- Build a small database of elephant IDs linked to their visual features.  

---

## 📌 Current Status
- Building a dataset of elephant videos (public sources, YouTube, wildlife repositories).  
- Designing a prototype pipeline that detects, tracks, and groups elephants per video.  
- Preparing feature extraction and comparison methods for re-identification.  

---

## 🔮 Future Work
- Improve accuracy by fine-tuning detection on elephant-specific datasets.  
- Train a dedicated elephant re-identification model.  
- Scale to handle large datasets for real-world deployment.  
- Explore edge deployment for use in camera traps or drones.  

---

## 👩‍💻 Authors
- Team Members : 
    Snehasis Das
    Md. Zafir Hasan
    Subhadeep Mandal
    Neelabhra Das

- Final Year Project, [Narula Institute Of Technology, Agarpara, Kolkata, West Bengal, India]