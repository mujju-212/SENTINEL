# Face Database Guide

How to enroll known persons, manage the database, and tune recognition quality.

---

## How the Face Database Works

SENTINEL stores face data as **image folders**, not embeddings:

```
database/
└── known_faces/
    ├── Alice/
    │   ├── alice_front.jpg
    │   ├── alice_side.jpg
    │   └── alice_smile.jpg
    └── Bob/
        ├── bob1.jpg
        └── bob2.jpg
```

When DeepFace runs recognition, it:
1. Computes an embedding for the person detected in the frame
2. Compares it against embeddings of every image in `known_faces/`
3. Returns the folder name (person name) of the closest match if within threshold

This means:
- **Adding a person = putting their photos in a folder**
- **Removing a person = deleting their folder**
- **No separate training step needed** — DeepFace works on raw images

---

## Enrolling a Person

### Method 1 — Interactive Tool (recommended)

```bash
python add_person.py
```

Follow the prompts:

```
==============================
  Face Enrollment Tool
==============================

Options:
  1. Import from folder of images
  2. Capture from webcam
  3. List enrolled persons
  4. Remove a person
  5. Exit

Enter choice: _
```

**Option 1 — Import from folder:**
- Point it at a folder containing the person's photos
- It copies them to `database/known_faces/<name>/`
- Validates each photo has a detectable face before adding
- Rejects blurry or faceless images

**Option 2 — Webcam capture:**
- Sit the person in front of the camera
- The tool captures 5 photos automatically
- Total time: ~15 seconds
- Saves to `database/known_faces/<name>/`

---

### Method 2 — Manual (drag and drop)

Simply create a folder with the person's name and put photos in it:

```
database/known_faces/
└── John_Smith/           ← Folder name becomes the display label
    ├── photo1.jpg
    ├── photo2.jpg
    └── photo3.jpg
```

Rules:
- Folder name = display label (shown in the HUD)
- Use underscores instead of spaces: `John_Smith` not `John Smith`
- Photos can be JPG or PNG
- Face must be visible and at least 80×80 pixels

---

## Photo Quality Guidelines

Recognition accuracy depends heavily on photo quality.

### Good photos

- Face clearly visible, unobstructed
- Varied angles: front, slight left, slight right
- Varied lighting: indoor, outdoor, different times of day
- Normal expression (multiple expressions adds robustness)
- At least 5–10 photos per person

### Bad photos (will reduce accuracy)

- Blurry or low resolution
- Strong backlit / heavy shadows
- Sunglasses, masks, or obstructions
- Face too small (less than ~80px height)
- Multiple people in the same photo *(tool will ask you to select which face)*

### Minimum recommended

| Scenario | Min Photos | Notes |
|---|---|---|
| Controlled environment | 3–5 | Good lighting, front-facing camera |
| Indoor variable lighting | 5–10 | Mix of angles and lights |
| Outdoor surveillance | 10–15 | High variation in lighting needed |

---

## Tuning Recognition

### Threshold

`recognition_threshold` in `config.yaml` controls how strict matching is:

| Value | Behavior |
|---|---|
| `0.40` | Very strict — only matches if nearly identical |
| `0.60` | Default — good balance |
| `0.80` | Lenient — higher false positives |

If you're getting:
- **False positives** (wrong person identified): lower the threshold (e.g. 0.50)
- **False negatives** (known person not identified): raise threshold (e.g. 0.70) or add more photos

### Detector backend

| Backend | Best for |
|---|---|
| `retinaface` | Small faces, surveillance cameras, best accuracy |
| `mtcnn` | Video streams, moderate distance |
| `opencv` | Close-up faces, fast CPU-only |

Change in `config.yaml`:
```yaml
face_recognition:
  detector_backend: "retinaface"
```

### Embedding model

| Model | Best for |
|---|---|
| `Facenet` | Speed + accuracy balance (default) |
| `Facenet512` | Better accuracy, slightly slower |
| `ArcFace` | Best accuracy for surveillance |

---

## Checking the Database

List all enrolled persons:

```bash
python add_person.py --list
```

Output:
```
Enrolled persons:
  Alice          5 photos
  Bob            8 photos
  Charlie        3 photos

Total: 3 persons, 16 photos
```

---

## Removing a Person

### Via the tool:
```bash
python add_person.py
# Choose option 4 → enter person name → confirm
```

### Manually:
```powershell
Remove-Item -Recurse "D:\AVTIVE PROJ\SENTINEL\database\known_faces\Alice"
```

---

## Backing Up the Face Database

The entire face database is just a folder of photos — easy to back up:

```powershell
# Backup
Copy-Item -Recurse "database\known_faces" "D:\Backups\known_faces_backup"

# Restore
Copy-Item -Recurse "D:\Backups\known_faces_backup" "database\known_faces"
```

> The `known_faces/` folder is excluded from git (`.gitignore`) — it contains private biometric data and should never be pushed to GitHub.

---

## How Person Logs Work (Database)

Every time a recognized person appears on camera, the `person_logs` table records:
- First time seen in this session
- Last time seen
- Total time in frame (seconds)
- Total visits (across all sessions)

Query via DB Browser for SQLite or Python:
```python
from utils.database import DatabaseManager
import yaml

config = yaml.safe_load(open('config.yaml'))
db = DatabaseManager(config['database'])

# Get today's person visits
stats = db.get_todays_stats()
print(stats)
```
