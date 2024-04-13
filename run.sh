# Build Backend
echo "Building Backend, this will take a while..."
cd api
docker build -t apibackend .
docker run -d -p 8000:8000 apibackend 
echo "Backend is running on port 8000"

# Build Frontend
echo "Building Frontend, this will take less time"
cd ../frontend
docker build -t apifrontend .
docker run -d -p 8080:8080 apifrontend
echo "Frontend is running on port 8080"