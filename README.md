# ChaiIntel - Project Setup Guide

This guide walks you through setting up the ChaiIntel Django project on your local machine.

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

### 🔹 On Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### 🔹 On macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
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


