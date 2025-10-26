### **1. Developing a Predictive Model for Automation Tasks**

```python
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
import smtplib 
from email.mime.text import MIMEText 
from email.mime.multipart import MIMEMultipart  

data = { 
    'age': [22, 35, 58, 45, 25, 33, 52, 40, 60, 48],
    'browsing_time': [5, 10, 2, 8, 7, 12, 1, 4, 3, 9],
    'previous_purchases': [0, 3, 1, 5, 0, 2, 0, 1, 0, 4],
    'made_purchase': [0, 1, 0, 1, 0, 1, 1, 1, 1, 1]
} 

df = pd.DataFrame(data)
X = df[['age', 'browsing_time', 'previous_purchases']]
y = df['made_purchase']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

new_customer = {'age': 35, 'browsing_time': 10, 'previous_purchases': 3, 'email': 'xxxx@gmail.com'}
new_df = pd.DataFrame([new_customer])  
prediction = model.predict(new_df[['age', 'browsing_time', 'previous_purchases']])
print("Prediction for new customer:", prediction[0])

if prediction[0] == 1: 
    sender_email = "yyyy@gmail.com" 
    sender_password = "hhprrnmnugkrihcf"
    receiver_email = new_customer['email'] 
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = "Special Offer Just for You!"
    body = """
    Hi there,
    We noticed you're interested in our products. Here's a special offer just for you!
    Best regards,
    Your Company
    """ 
    message.attach(MIMEText(body, "plain"))
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_email, message.as_string())
        print("Email sent successfully to", receiver_email)
    except Exception as e:
        print("Failed to send email:", e)
    finally:
        server.quit()
else:
    print("Customer unlikely to purchase — no email sent.")
```

---

### **2. Weather Data Collection and Analysis**

```python
import requests

def get_weather(city):
    url = f"https://wttr.in/{city}?format=3"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print(f"Current weather in {city}: {response.text}")
        else:
            print("Failed to get weather info")
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    city = "Chennai"
    print(f"--- Weather Information for {city} ---")
    get_weather(city)
```

---

### **3. Automated Data Preprocessing Techniques**

```python
import pandas as pd

df = pd.read_excel("C:/Users/User/Desktop/bike.xlxs")
print("Original Data:")
print(df.head())

df = df.drop_duplicates()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].str.strip().str.lower()

df['age'] = df['age'].fillna(df['age'].mean())
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date'])

print("\nCleaned Data:")
print(df.head())

df.to_csv("data_cleaned.csv", index=False)
```

---

### **4. Process Automation Strategies**

```python
import pandas as pd
import smtplib
from email.mime.text import MIMEText

data = pd.read_excel("/content/drive/MyDrive/employee.xlsx")
Absent = data[data['Status'] == 'Absent']

sender_email = "integratedaidnnn@gmail.com"
password = "hhprrnmnugkrihcf"

server = smtplib.SMTP("smtp.gmail.com", 587)
server.starttls()
server.login(sender_email, password)

for _, row in Absent.iterrows():
    name = row['Name']
    recipient = row['Email']
    msg = MIMEText("ABSENT")
    msg['Subject'] = "Absent Message"
    msg['From'] = sender_email
    msg['To'] = recipient
    server.sendmail(sender_email, recipient, msg.as_string())
    print(f"Email sent to {name}")

server.quit()
print("Automation completed!")
```

---

### **5. Introduction to Web Scraping Methods**

```python
import requests
from bs4 import BeautifulSoup

url = "https://news.ycombinator.com/"
headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(url, headers=headers)

if response.status_code == 200:
    soup = BeautifulSoup(response.text, "html.parser")
    title_spans = soup.find_all("span", class_="titleline")
    print("Top Headlines:")
    for i, span in enumerate(title_spans[:5], start=1):
        link = span.find("a")
        print(f"{i}. {link.text}")
else:
    print("Failed to fetch page. Status code:", response.status_code)
```

---

### **6. Advanced Web Scraping Techniques**

```python
import requests
from bs4 import BeautifulSoup

url = "https://books.toscrape.com/catalogue/category/books_1/index.html"
headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(url, headers=headers)

if response.status_code == 200:
    soup = BeautifulSoup(response.text, "html.parser")
    books = soup.find_all("h3")
    prices = soup.find_all("p", class_="price_color")

    for i, (book, price) in enumerate(zip(books, prices), start=1):
        print(f"{i}. {book.a['title']} – {price.text}")
else:
    print(f"Failed to fetch page. Status code: {response.status_code}")
```

---

### **7. Robotic Process Automation (RPA)**

```python
import requests
from bs4 import BeautifulSoup
import pandas as pd

all_books = []

for page in range(1, 51):
    url = f"https://books.toscrape.com/catalogue/page-{page}.html"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"Failed to fetch page {page}. Status code: {response.status_code}")
        break

    soup = BeautifulSoup(response.text, "html.parser")
    books = soup.find_all("h3")
    prices = soup.find_all("p", class_="price_color")

    for book, price in zip(books, prices):
        all_books.append({
            "Title": book.a['title'],
            "Price": price.text
        })

df = pd.DataFrame(all_books)
df.to_csv("books_data.csv", index=False, encoding='utf-8')
print("Scraping complete!")
print(f"Total books collected: {len(df)}")
print(df.head())
```

---

### **8 (A & B). Python Libraries for Automation**

#### **A. Scheduling Tasks Using the Schedule Library**

```python
import schedule
import time

def job():
    print("Reminder: Take a short break and stretch!")

schedule.every(5).seconds.do(job)
print("Scheduler started. Press Ctrl+C to stop.")

while True:
    schedule.run_pending()
    time.sleep(1)
```

#### **B. Automated Invoice Generation**

```python
from fpdf import FPDF

purchase_items = [
    {"item": "Rice Bag", "qty": 2, "price": 1200},
    {"item": "Sugar", "qty": 5, "price": 45},
    {"item": "Oil Tin", "qty": 1, "price": 1500},
    {"item": "Milk Packet", "qty": 10, "price": 25},
]

def generate_invoice(invoice_no, customer_name, purchases):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, txt="INVOICE", ln=True, align="C")

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Invoice No: {invoice_no}", ln=True)
    pdf.cell(200, 10, txt=f"Customer: {customer_name}", ln=True)
    pdf.ln(10)

    pdf.cell(80, 10, "Item", 1)
    pdf.cell(30, 10, "Qty", 1)
    pdf.cell(40, 10, "Price", 1)
    pdf.cell(40, 10, "Amount", 1, ln=True)

    total = 0
    for p in purchases:
        amount = p["qty"] * p["price"]
        total += amount
        pdf.cell(80, 10, p["item"], 1)
        pdf.cell(30, 10, str(p["qty"]), 1)
        pdf.cell(40, 10, f"{p['price']}", 1)
        pdf.cell(40, 10, f"{amount}", 1, ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(150, 10, "TOTAL", 1)
    pdf.cell(40, 10, f"{total}", 1, ln=True)

    filename = f"Invoice_{invoice_no}.pdf"
    pdf.output(filename)
    print(f"Invoice saved as {filename}")
```

---

### **9. Cognitive Automation with NLP: Email Classification Applications**

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

data = {
    'email': [
        'I want a refund, the product is defective',
        'Your service is excellent, thank you!',
        'Where is my order? It’s late again!',
        'Please cancel my subscription immediately',
        'This is spam. Stop sending these emails!',
        'I need help with my account login issue',
        'Thanks for the quick response, great support',
        'You guys are scammers, I lost my money!',
        'My package was damaged during delivery',
        'Congratulations! You won a free iPhone!'
    ],
    'label': [
        'complaint', 'praise', 'complaint', 'support_request',
        'spam', 'support_request', 'praise', 'complaint',
        'complaint', 'spam'
    ]
}

df = pd.DataFrame(data)
X_train, X_test, y_train, y_test = train_test_split(df['email'], df['label'], test_size=0.3, random_state=42)
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))

new_email = "Why did you charge me twice for the same item?"
predicted_label = model.predict([new_email])[0]
print(f'\nNew Email: "{new_email}"\nPredicted Category: {predicted_label}')
```

---

Would you like me to **extract and save** all these 9 programs as a **single `.py` file** (clean and formatted)?
