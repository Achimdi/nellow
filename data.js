/* ALL SITE CONTENT LIVES HERE. Edit the text, save, refresh. Anything in [BRACKETS] is a placeholder that shows with a dashed outline. */
const SITE = {
  name: "Azie Chimdiume",
  status: "Open to AI/ML roles",
  photo: "assets/photo.jpg",
  hero: {
    headline: "I build AI that shows its work.",
    sub: "Explainable intelligent systems, end to end: from deep learning for medical imaging to autonomous AI agents, with the interface and backend to ship them."
  },
  links: {
    github: "https://github.com/Achimdi",
    linkedin: "https://www.linkedin.com/in/chimdiume-azie-73a477448/",
    instagram: "",                                  // add your Instagram link, or leave empty to hide it
    email: "[PLACEHOLDER: your email address]",
    cv: "[PLACEHOLDER: path to your CV file, for example assets/cv.pdf]"
  },
  alz: {
    title: "Alzheimer's Disease Detection with Deep Learning and Explainable AI",
    status: "Working research prototype. Not clinically validated. Not a medical device.",
    problem: "Alzheimer's is often diagnosed late. Hospitals in Nigeria have few specialists and limited resources, manual MRI reading is slow and subjective, and most AI research stops at a prediction with no explanation and no clinical workflow around it.",
    parts: [
      ["Classifier", "A deep learning model that sorts brain MRI scans into four stages: NonDemented, VeryMildDemented, MildDemented and ModerateDemented."],
      ["Explainability", "Grad-CAM overlays a heatmap on the scan to show which regions drove each prediction. For Mild and Moderate predictions the heatmaps concentrated on the medial temporal lobe, hippocampus and ventricles, regions known to be affected by Alzheimer's."],
      ["Hospital workflow", "A web app with Admin and Doctor roles: JWT authentication and role-based access control, one-time doctor registration codes issued by the admin, patient records, MRI upload that returns the stage, confidence score and heatmap, stored result history, and patient transfers between doctors with a full audit trail. SQLite through SQLAlchemy, 6 tables."]
    ],
    spec: [
      ["Model", "EfficientNet-B0, ImageNet pretrained (timm)"],
      ["Training", "Transfer learning, all 4,012,672 parameters fine-tuned"],
      ["Input", "224 x 224"],
      ["Augmentation", "Horizontal flip, rotation up to 10 degrees, brightness and contrast jitter"],
      ["Imbalance", "Class-weighted cross-entropy (ModerateDemented weight 13.312)"],
      ["Optimiser", "Adam, learning rate 0.0001, batch 32, 25 epochs, step schedule"],
      ["Data", "OASIS Alzheimer's MRI (Kaggle): 35,484 train, 8,871 validation, 46,184 test"],
      ["Hardware", "Ordinary Windows CPU laptop, no GPU, about 119 hours of training, chosen on purpose for limited hospital hardware"]
    ],
    val: 98.66, valNote: "Validation accuracy at its best (epoch 19), up from 84.84% after epoch 1. Competitive with the published four-class and three-class results I reviewed (97.77% to 99.21%).",
    test: 77.35, testNote: "Held-out test accuracy on 46,184 images. Weighted precision 0.8726, recall 0.7735, F1 0.8126.",
    diagnosis: "The gap is a dataset distribution shift that I diagnosed myself. The original test split was 88.5% NonDemented against 40.8% in training. A model that always answered NonDemented would score about 88.5% on it, so accuracy alone was misleading. The minority classes paid for it: ModerateDemented had only 168 test samples and scored 0 precision and recall.",
    next: ["Stratified splitting", "GPU training", "Local Nigerian hospital data", "Validation with practising neurologists and radiologists", "Tracking a patient's progression across scans", "A lightweight offline and mobile version, for example MobileNetV3"],
    stages: [
      { k: "NonDemented", p: "0.9550", r: "0.8100", note: "88.5% of the original test set.", img: "assets/stage-nondemented.jpg" },
      { k: "VeryMildDemented", p: "[add from document]", r: "0.5463", note: "Recall 0.5463: about half of these scans were found.", img: "assets/stage-verymild.jpg" },
      { k: "MildDemented", p: "0.0671", r: "0.2610", note: "Heatmaps concentrated on the medial temporal lobe and hippocampus.", img: "assets/stage-mild.jpg" },
      { k: "ModerateDemented", p: "0.0000", r: "0.0000", note: "Only 168 test samples, and none were found. This is the failure I diagnosed.", img: "assets/stage-moderate.jpg" }
    ],
    stack: ["Python 3.12", "PyTorch", "timm", "torchvision", "OpenCV", "scikit-learn", "NumPy", "Pandas", "Matplotlib", "Seaborn", "Flask", "Flask-JWT-Extended", "SQLAlchemy", "HTML/CSS/JavaScript"],
    role: "Everything: data pipeline, model, explainability, backend and interface.",
    repo: "[PLACEHOLDER: link to the Alzheimer's project repository]",
    demo: "[PLACEHOLDER: demo link, or delete this line]"
  },
  nellow: {
    title: "Nellow, an AI agent I built",
    status: "Runs locally on my laptop. Hosting is planned.",
    summary: "An AI agent that searches the live web, reads what it finds and answers with source links. It has saved chats, projects, accounts, image understanding and voice, and a futuristic metallic interface with 3D motion.",
    features: ["Live web and news search with source links", "Three modes: Balanced runs on the device, Thorough and Turbo run in the cloud", "Image understanding, voice dictation and read-aloud replies", "Accounts with emailed verification codes, saved chats and projects"],
    stack: ["Python", "FastAPI", "SQLite", "Ollama", "Groq API", "JavaScript", "Three.js"],
    role: "Designed and built end to end: agent logic, backend, accounts and interface.",
    repo: "https://github.com/Achimdi/nellow",
    demo: "[PLACEHOLDER: live demo link once it is hosted, or delete this line]"
  },
  about: [
    "I'm a recent Computer Science graduate who builds AI that people can check. My final-year project taught me that a prediction only helps when someone can see why the model said it, and where it fails.",
    "I built it on a CPU laptop on purpose, because hospitals with limited resources cannot assume a GPU server. And I report my results in full, including the ones that went against me, because that is how the work gets better.",
    "Now I'm looking for my first AI/ML or software role, where I can keep building intelligent systems end to end."
  ],
  skills: [
    { g: "Machine learning and AI", d: .9, items: ["PyTorch", "Transfer learning", "CNNs", "Grad-CAM / explainable AI", "scikit-learn", "OpenCV", "NumPy", "Pandas", "Precision, recall, F1", "Confusion matrix and ROC", "Class imbalance"] },
    { g: "Web and backend", d: .8, items: ["HTML", "CSS", "JavaScript", "Flask", "REST APIs", "JWT auth and role-based access", "SQLite / SQLAlchemy"] },
    { g: "Mobile", d: .45, items: ["Flutter (working knowledge)"] },
    { g: "Tools", d: .75, items: ["Python", "Git / GitHub", "VS Code"] }
  ],
  path: [
    { when: "[PLACEHOLDER: graduation year]", what: "Computer Science, Godfrey Okoye University", more: "[PLACEHOLDER: degree title]. Coursework: Expert Systems, Computer Networks and Communication, Compiler Construction, Structured Programming, Human Computer Interaction, Systems Analysis and Design." },
    { when: "[PLACEHOLDER: dates]", what: "[PLACEHOLDER: internship or work experience, or delete this entry]", more: "" }
  ]
};
