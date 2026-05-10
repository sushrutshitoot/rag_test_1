# Use an official Python 3.13 slim image
FROM python:3.13-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies (including uv for faster installs if desired, but pip works fine in Docker)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY src/ ./src/

# Copy the local databases into the container so it's baked into the image
# (For production, you might want to mount this via volumes instead)
COPY chroma_db ./chroma_db
COPY docstore ./docstore

# Expose the Streamlit port
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "src/rag_app/ui/app.py", "--server.address=0.0.0.0"]
