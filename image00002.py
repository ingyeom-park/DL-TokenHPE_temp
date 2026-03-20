from scipy import io

path = "image00002.mat"

# 데이터 파일 불러오기
mat_file = io.loadmat(path)
print(mat_file)