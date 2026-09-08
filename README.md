# 🎵 Moosic - ML-Powered Playlist Generator

> **Can machines create playlists that *feel* right?**
>
> This project uses unsupervised machine learning to automatically generate mood-based playlists from Spotify audio data. 

---

## 🌍 Why This Matters Beyond Music

The core idea behind this project - **grouping things by similarity using data** - applies far beyond playlists:

| Industry | Application | How It Works |
|----------|------------|--------------|
| 🛒 **Retail** | Customer segmentation | Group shoppers by purchase behavior → targeted marketing campaigns |
| 🏥 **Healthcare** | Patient clustering | Group patients by symptoms and lab results → personalized treatment paths |
| 📰 **Media** | Content recommendation | Group articles/videos by topic and tone → "You might also like..." |
| 🏦 **Finance** | Fraud detection | Group transactions by patterns → flag unusual clusters as potential fraud |
| 🎓 **Education** | Student grouping | Group learners by performance patterns → adaptive learning paths |
| 🏨 **Hospitality** | Guest profiling | Group hotel guests by preferences → personalized room/service offers |

**The same algorithm that groups songs into playlists can group customers into segments, patients into risk categories, or products into recommendation clusters.** The technical skills are transferable - only the data changes.

---

## 📖 The Story (Non-Technical)

**The problem:** Moosic is a startup that creates curated playlists by music experts. Business is growing fast, but experts are slow. Can technology help?

**What we did:** We took 5,235 songs from Spotify. Each song comes with numerical characteristics - how danceable it is, how much energy it has, whether it sounds happy or sad, acoustic or electronic. We fed these numbers to a machine learning algorithm and asked it: *"Group these songs into playlists where songs sound similar and share a similar mood."*

**How it works (in plain English):**

1. **Step 1 - Find musical families:** The algorithm analyzed all 5,235 songs and discovered 8 natural groups based on how similar they sound overall. Think of it as sorting a music store into 8 sections.

2. **Step 2 - Split by mood:** Each group was still too large (600+ songs), so we created a "mood score" for every song - based on its energy, happiness, and acoustic character - and split each group into smaller playlists of ~200 songs. Songs with similar mood scores ended up together.

**The result:** 27 playlists, each containing 152–213 songs. A playlist of Brazilian bossa nova. A playlist of classical piano and ambient electronic. A playlist of upbeat pop and dance. All created automatically. We listened to some songs to evaluate the results.

**Is it perfect?** No. The algorithm sometimes puts a death metal track next to Chopin because they share similar "sadness" scores - it sees numbers, not genres. But as a first prototype, it proves the concept works and is worth developing further.

---

## 🧠 Machine Learning Skills Demonstrated

### Unsupervised Learning
- **K-Means Clustering** - partitioning data into K groups by minimizing within-cluster distances
- Understanding the difference between supervised (learning from labeled examples) and unsupervised (discovering structure without labels)
- Working with unlabeled data where there is no "correct answer"

### Feature Engineering & Selection
- Selecting meaningful features from 18 available columns (kept 9 audio features, dropped categorical and irrelevant ones)
- Understanding *why* certain features don't work with distance-based algorithms (categorical variables like `key` and `mode`)
- Using domain knowledge (Russell's Circumplex Model from psychology) to create a composite **mood score** from energy, valence, and acousticness

### Data Preprocessing
- **Feature scaling** (MinMaxScaler, StandardScaler) - understanding why distance-based algorithms require scaled data
- Comparing scaling methods and evaluating their impact on clustering results
- Handling data quality issues (missing values, duplicates, trailing whitespace in column names)

### Model Evaluation
- **Elbow Method** - plotting inertia across K values to find the optimal number of clusters
- **Silhouette Score** - measuring cluster cohesion and separation (-1 to +1)
- Balancing mathematical metrics with business constraints (50–250 songs per playlist)
- Qualitative evaluation - inspecting actual playlist contents to assess whether clusters make sense

### Scikit-Learn Proficiency
- The `Import → Initialize → Fit → Transform` pattern used across all Scikit-Learn tools
- `KMeans`: hyperparameters (`n_clusters`, `random_state`, `n_init`), attributes (`.labels_`, `.cluster_centers_`, `.inertia_`)
- `MinMaxScaler` and `StandardScaler`: understanding when to use each
- `silhouette_score` for evaluation

### Data Visualization
- Distribution plots, correlation heatmaps, radar charts (Plotly), scatter plots
- Choosing the right visualization for the audience (radar charts for CEO, heatmaps for technical review)
- Custom color palettes aligned with brand identity

### Business Thinking
- Translating a business problem ("create playlists") into a data science problem ("cluster songs by audio similarity")
- Navigating the tension between mathematical optimality and business requirements
- Communicating technical results to non-technical stakeholders
- Proposing actionable next steps (A/B testing, data enrichment, user feedback loops)

---

## 🔬 Methodology

### The 2-Step Approach

```
5,235 Spotify songs
    │
    ▼  Step 1: K-Means Clustering (K=8, all 9 audio features, MinMaxScaler)
    │  → 8 broad clusters grouped by overall musical similarity
    │
    ▼  Step 2: Mood Score Splitting
    │  → mood_score = mean(valence, energy, acousticness)
    │  → within each cluster, sort by mood score and split into ~200-song playlists
    │
    ▼  Result: 27 playlists of 152–213 songs each ✅
```

### Why 2 Steps?

- **K-Means with large K (20+):** Poor silhouette scores, uneven cluster sizes
- **K-Means with small K (8):** Good cluster quality (silhouette: 0.2778), but clusters too large (~600 songs)
- **Solution:** Small K for quality clustering → mood-based split for business-appropriate sizes

### Audio Features Used

| Feature | Range | What It Measures |
|---------|-------|-----------------|
| `danceability` | 0–1 | Suitability for dancing (tempo, rhythm, beat strength) |
| `energy` | 0–1 | Intensity and activity (death metal = high, Bach prelude = low) |
| `valence` | 0–1 | Musical positiveness (happy = high, sad = low) |
| `acousticness` | 0–1 | Likelihood of being acoustic vs. electronic |
| `instrumentalness` | 0–1 | Likelihood of containing no vocals |
| `speechiness` | 0–1 | Presence of spoken words |
| `liveness` | 0–1 | Probability of live recording |
| `loudness` | -60–0 dB | Average volume in decibels |
| `tempo` | BPM | Speed in beats per minute |

### The Mood Score (Russell's Circumplex Model)

The mood score is inspired by [Russell's Circumplex Model of Affect](https://en.wikipedia.org/wiki/Emotion_classification#Circumplex_model) (1980), which maps emotions on two axes:

- **Energy → Arousal** (calm ↔ intense)
- **Valence → Valence** (sad ↔ happy)
- **Acousticness → Corrective factor** (distinguishes acoustic ballads from electronic ambient)

```
                    High Energy
                        │
         😤 Intense     │     😄 Happy
         (rock, metal)  │     (pop, dance)
                        │
   Low Valence ─────────┼───────── High Valence
                        │
         😢 Sad         │     😌 Relaxed
         (blues, ballad)│     (chill, classical)
                        │
                    Low Energy
```

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| Dataset size | 5,235 songs |
| Features used | 9 audio features |
| Step 1 clusters | 8 (K-Means) |
| Final playlists | 27 |
| Smallest playlist | 152 songs |
| Largest playlist | 213 songs |
| Silhouette score (K=8) | 0.2778 |
| Scaling method | MinMaxScaler |

### What Worked
- ✅ All playlists within 50–250 business requirement
- ✅ Playlist 1_1: Brazilian bossa nova grouped perfectly (Marcos Valle, Seu Jorge, João Gilberto)
- ✅ Cluster 0: Classical piano + ambient (Chopin, Debussy, Brian Eno, Max Richter)

### What Didn't
- ⚠️ Playlist 0_0 includes 2–3 death metal tracks among classical - same low valence, completely different genre
- ⚠️ Audio features don't capture genre, lyrics language, or cultural context

---

## 📂 Repository Structure

```
moosic-ml-playlist-generator/
│
├── moosic_github_final.ipynb    # Main analysis notebook
├── 3_spotify_5000_songs.csv     # Dataset (5,235 songs with audio features)
├── Moosic_Presentation.pdf      # Stakeholder presentation (5 min)
└── README.md                    # This file
```

---

## 🛠 Tech Stack

- **Python** - primary language
- **Pandas & NumPy** - data manipulation
- **Scikit-Learn** - KMeans, MinMaxScaler, StandardScaler, silhouette_score
- **Matplotlib & Seaborn** - static visualizations
- **Plotly** - interactive radar charts
- **Jupyter Notebook** - analysis environment

---

## 🚀 Next Steps

1. **A/B test** ML playlists vs. expert-curated with real subscribers
2. **Enrich data** with genre labels and lyrics analysis (NLP)
3. **User feedback loop** - let subscribers rate playlists to improve over time
4. **Explore alternative algorithms** - DBSCAN (no K needed), Hierarchical Clustering (dendrogram)
5. **Hybrid approach** - algorithm creates first draft, music experts add the finishing touch

---

## 🎓 Context

This project was completed as part of the **Unsupervised Machine Learning** module at [WBS Coding School](https://www.wbscodingschool.com/) Data Science Bootcamp.

**The brief:** Moosic, a playlist curation startup, hired us to determine whether machine learning could automate playlist creation using Spotify's audio features. Our task was to build a prototype, evaluate it, and present our findings to the (simulated) startup team.

---

## 📎 Presentation

The stakeholder presentation (5 minutes, non-technical audience) is available as a PDF in this repository:

📄 **[ML_Powered_Playlist_Generator.pdf](./ML_Powered_Playlist_Generator.pdf)**

---
