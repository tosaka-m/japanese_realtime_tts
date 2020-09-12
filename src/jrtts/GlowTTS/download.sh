fileId=1JiCMBVTG4BMREK8cT3MYck1MgYvwASL0
mkdir -p logs/pretrained_models
outPath=logs/pretrained_models/pretrained.pth
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${fileId}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${fileId}" -o ${outPath}
