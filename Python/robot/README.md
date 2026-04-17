# 🎓 AI-Based Attendance System (Face Recognition Model)

This repository contains the Machine Learning module for a Graduation Project 2026.

The system is designed to perform **real-time face recognition using ArcFace** in order to build a smart attendance system based on webcam input.

---

## 🚀 Project Overview

This module handles:

- Face Detection
- Face Alignment
- Face Embedding Extraction (ArcFace)
- Real-Time Face Recognition
- Embedding Database Generation

The goal is to provide a stable and scalable recognition engine that can later be integrated with a backend attendance system.

---

## 🧠 Model Architecture

The system uses:

- **InsightFace (buffalo_l model)**
- **ArcFace (w600k_r50)**
- Cosine Similarity for matching
- GPU acceleration (CUDAExecutionProvider)

Pipeline:

Webcam Frame  
→ Face Detection  
→ Face Alignment  
→ Embedding Extraction  
→ Compare with Stored Embeddings  
→ Identity Decision  
