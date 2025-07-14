# S3 Bucket 設置說明

## 1. 創建 S3 Bucket

```bash
# 使用 AWS CLI 創建 bucket
aws s3 mb s3://church-highlights-videos --region us-east-1
```

## 2. 設置 Public Read 權限

創建 bucket policy，允許公共讀取：

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "PublicReadGetObject",
            "Effect": "Allow",
            "Principal": "*",
            "Action": "s3:GetObject",
            "Resource": "arn:aws:s3:::church-highlights-videos/*"
        }
    ]
}
```

## 3. 設置 CORS 配置

```json
[
    {
        "AllowedHeaders": ["*"],
        "AllowedMethods": ["GET", "HEAD"],
        "AllowedOrigins": ["*"],
        "ExposeHeaders": ["ETag"],
        "MaxAgeSeconds": 3000
    }
]
```

## 4. 環境變量設置

複製 `env.example` 到 `.env` 並填入您的 AWS 憑證：

```bash
cp env.example .env
```

編輯 `.env` 文件：

```env
# AWS S3 配置
AWS_ACCESS_KEY_ID=your_aws_access_key_here
AWS_SECRET_ACCESS_KEY=your_aws_secret_key_here
AWS_REGION=us-east-1
S3_BUCKET_NAME=church-highlights-videos
```

## 5. 上傳視頻到 S3

```bash
# 上傳單個視頻
aws s3 cp "video.mp4" s3://church-highlights-videos/

# 上傳整個目錄
aws s3 sync ./videos/ s3://church-highlights-videos/
```

## 6. 測試訪問

確保可以通過公共 URL 訪問視頻：

```
https://church-highlights-videos.s3.us-east-1.amazonaws.com/your-video.mp4
```

## 7. 目錄結構建議

```
church-highlights-videos/
├── sermons/
│   ├── 2024/
│   │   ├── 01-january/
│   │   └── 02-february/
│   └── 2025/
│       └── 01-january/
├── events/
└── testimonies/
```

## 8. 重要提醒

- 確保 bucket 名稱是唯一的
- 設置適當的 lifecycle 規則管理存儲成本
- 定期檢查和清理不需要的文件
- 考慮使用 CloudFront 作為 CDN 加速視頻加載