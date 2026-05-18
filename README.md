# ChaiIntel - Project Setup Guide (Windows)

This guide walks you through setting up the ChaiIntel Django project on a Windows machine.

---

## 🚀 Quick Start (One Command)

If you just want to run the application and you are **not a programmer**, follow these three steps:

1. Install **Python 3.9+** from [python.org](https://www.python.org/downloads/) and make sure to tick **"Add Python to PATH"** during installation.
2. Open the `ChaiIntel` folder in File Explorer.
3. **Double-click `run_chaiintel.bat`**.

That single script will:

* create the virtual environment (the first time only),
* activate it,
* install all required libraries (only when `requirements.txt` changes),
* apply database migrations, and
* start the application at [http://127.0.0.1:8000](http://127.0.0.1:8000).

When you see *"Starting development server at http://127.0.0.1:8000"*, open that link in your browser. To stop the app, press **CTRL+C** in the black window or just close it.

### All-in-one terminal command

Prefer the terminal? From inside the `ChaiIntel` folder, run:

```bat
run_chaiintel.bat
```

---

## 🛠 Manual Setup (for developers)

The sections below explain what the launcher does, step by step, in case you want to set things up manually.

---

## ✅ Step 1: Install Python

1. Download Python 3.9 or newer from [python.org](https://www.python.org/downloads/).
2. During installation, check the box:

   * ✅ *Add Python to PATH*

---

## ✅ Step 2: Install Git

1. Download Git from [git-scm.com](https://git-scm.com/downloads).
2. Install it with default settings.
3. Verify installation:

   ```bash
   git --version
   ```

---

## ✅ Step 3: Clone the Repository

```bash
git clone https://github.com/benodongo/ChaiIntel.git
cd ChaiIntel
```

---

## ✅ Step 4: Create Virtual Environment

```bat
python -m venv venv
venv\Scripts\activate
```

---

## ✅ Step 5: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---



## ✅ Step 6: Apply Migrations

```bash
python manage.py migrate
```

---

## ✅ Step 7: Run Development Server

```bash
python manage.py runserver
```

Visit: [http://127.0.0.1:8000](http://127.0.0.1:8000)

---


