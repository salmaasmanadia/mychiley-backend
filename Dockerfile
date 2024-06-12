# Gunakan image Python versi terbaru sebagai dasar
FROM python:3.9-slim

# Set working directory di dalam container
WORKDIR /app

# Copy file requirements.txt ke dalam container
COPY requirements.txt .

# Install dependensi Python yang diperlukan
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh konten proyek ke dalam container
COPY . .

# Expose port yang digunakan oleh aplikasi Flask
EXPOSE 5000

# Command untuk menjalankan aplikasi ketika container berjalan
CMD ["python", "app.py"]
