Setting up.
Navigate to your project directory
Open your terminal or command prompt, then:
cd path/to/your/project
Create the virtual environment
python -m venv venv

 Activate the virtual environment
 venv\Scripts\activate

 Install packages from requirements.txt
 pip install -r requirements.txt

 Apply migrations
 python manage.py makemigrations
 python manage.py migrate

 Run the development server
 python manage.py runserver
