# Setup Guide — from zip file to a working, version-controlled project

Follow this once, in order. It assumes you're on the computer where you want the
project to live (the one with the Desktop folder), with Python 3.10+ and Git
already installed. If you don't have Git: https://git-scm.com/downloads

---

## 1. Unpack the project onto your Desktop

You were given `clearsar.zip`. Unzip it so the result is:

```
Desktop/
└── clearsar/
    ├── README.md
    ├── docs/
    ├── src/
    └── ...
```

**Windows (PowerShell):**
```powershell
cd $HOME\Desktop
Expand-Archive clearsar.zip -DestinationPath .
cd clearsar
```

**macOS / Linux (Terminal):**
```bash
cd ~/Desktop
unzip clearsar.zip
cd clearsar
```

---

## 2. Create a Python environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If `rasterio` or `s2cloudless` fail to install on Windows, use conda instead for
those two packages (`conda install -c conda-forge rasterio s2cloudless`), then
`pip install` the rest.

---

## 3. Turn the folder into a git repository

```bash
git init
git add .
git commit -m "Initial project scaffold: plan, scene-finder script, folder structure"
```

Check it worked:
```bash
git log --oneline
```
You should see one commit.

**Why `git add .` is safe here:** the `.gitignore` already excludes large/private
things (`data/raw/`, model checkpoints, your virtual environment, `kaggle.json`).
Run `git status` before committing each time to eyeball what's about to be added —
if you ever see a `.tif`, a checkpoint, or `kaggle.json` listed, stop and check
`.gitignore` before committing.

---

## 4. Create the GitHub repository

**Option A — GitHub CLI (`gh`), if installed:**
```bash
gh auth login          # one-time, follow the prompts
gh repo create clearsar --private --source=. --remote=origin --push
```
This single command creates the repo on GitHub, links it as `origin`, and pushes.
Skip to step 6.

**Option B — no CLI, use the website:**
1. Go to https://github.com/new
2. Repository name: `clearsar`
3. Visibility: **Private** (recommended while it's coursework-in-progress; make
   it public later if you want to show it off)
4. **Do not** tick "Add a README" or "Add .gitignore" — you already have both.
   Ticking them creates conflicting files and complicates the first push.
5. Click **Create repository**. GitHub shows you a page with commands — ignore
   it and continue to step 5 below (you already ran `git init`).

---

## 5. Connect your local repo to GitHub and push (Option B only)

GitHub will show your repository URL, e.g. `https://github.com/<your-username>/clearsar.git`

```bash
git branch -M main
git remote add origin https://github.com/<your-username>/clearsar.git
git push -u origin main
```

If prompted for a password: GitHub no longer accepts your account password over
HTTPS. Use a **Personal Access Token** instead:
GitHub → Settings → Developer settings → Personal access tokens → Generate new
token (classic) → scope `repo` → paste it in place of your password when asked.
(Or set up SSH keys instead, if you prefer — GitHub's docs:
https://docs.github.com/en/authentication/connecting-to-github-with-ssh)

---

## 6. Confirm it worked

```bash
git remote -v     # should show origin ...clearsar.git (fetch) and (push)
```
Refresh the GitHub page in your browser — you should see all the files.

---

## 7. Your weekly workflow from here on

Every time you finish a meaningful chunk of work (a working script, a trained
model, a finished experiment):

```bash
git add .
git status                     # double-check nothing huge/private is staged
git commit -m "Week 2: U-Net trained, overfit test passed"
git push
```

Commit **often and in small pieces** — e.g. "Added PSNR/SSIM/SAM metric functions
with unit tests" rather than one giant commit at the end of each week. Small
commits make it far easier to find *when* something broke, which is itself a
useful debugging habit for deep learning work.

**Update the checklist in `README.md`** as you complete each week — tick the
boxes. It costs 10 seconds and gives you (and anyone else looking at the repo) an
instant sense of progress.

**Never commit:**
- raw `.tif` data (`data/raw/` is already ignored)
- model checkpoints (`experiments/checkpoints/` is already ignored)
- `kaggle.json` or any API keys/tokens

**Always commit:**
- code changes (`src/`)
- the plan and any notes (`docs/`)
- small result artefacts: metrics CSVs, final plots, the results tables
  (`experiments/logs/`, `experiments/results/`)

---

## 8. Connecting to Kaggle (where the actual training happens)

You won't run training locally (4 GB VRAM isn't enough — see `docs/project_plan.md`).
The usual flow:

1. **Upload your code to Kaggle, not just data.** Easiest approach: push to GitHub
   (steps above), then in a Kaggle notebook, add a cell:
   ```python
   !git clone https://github.com/<your-username>/clearsar.git
   %cd clearsar
   !pip install -r requirements.txt -q
   ```
   This keeps Kaggle notebooks thin — they just pull your latest code and run it,
   rather than containing copy-pasted logic that drifts out of sync with git.
2. **Upload extracted data as a private Kaggle Dataset** (Kaggle → Datasets → New
   Dataset), then add it to your notebook as an input. Don't re-run the full
   `extract` step every session — extract once, save as a dataset, reuse it.
3. **Save checkpoints back out.** Kaggle notebooks auto-save `/kaggle/working/` up
   to 20 GB when you "Save & Run All". Download the checkpoint/metrics files you
   want to keep and drop them into `experiments/checkpoints/` or
   `experiments/logs/` locally, then commit the logs (not the checkpoints) to git.

---

## 9. If something goes wrong

- **"fatal: not a git repository"** → you're not inside the `clearsar` folder.
  `cd` into it first.
- **Accidentally committed something huge (e.g. a `.tif` slipped through):**
  ```bash
  git rm --cached path/to/file
  echo "path/to/file" >> .gitignore
  git commit -m "Remove accidentally committed data file"
  ```
  (If it was already pushed and you need it fully gone from history, that's a
  separate, more careful operation — ask me when you hit it, rather than guessing.)
- **Merge conflicts / diverged branches:** if you only ever work from one
  computer, you shouldn't hit this. If you do work from two, always `git pull`
  before you start a session and `git push` before you stop.
