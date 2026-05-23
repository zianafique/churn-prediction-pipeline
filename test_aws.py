# Import AWS SDK
import boto3

# Create S3 client
# This automatically finds credentials from ~/.aws/credentials
s3_client = boto3.client("s3", region_name="us-east-1")

# List all your S3 buckets
print("📦 Your S3 Buckets:")
response = s3_client.list_buckets()

# Loop through each bucket and print name
for bucket in response["Buckets"]:
    print(f"  - {bucket['Name']}")

# If no buckets, you'll see empty list
if not response["Buckets"]:
    print("  (No buckets found - did you create one in Step 1?)")

print("\n✅ AWS connection successful!")
# Create a test file and upload to S3
test_data = "Hello from Python! This is a test file."

# Upload to your bucket
s3_client.put_object(
    Bucket="churn-prediction-models-zian",
    Key="test-data.txt",
    Body=test_data,
)

print("\n📤 Test file uploaded to S3!")

# List files in bucket
objects = s3_client.list_objects_v2(Bucket="churn-prediction-models-zian")
print("\n📋 Files in bucket:")
if "Contents" in objects:
    for obj in objects["Contents"]:
        print(f"  - {obj['Key']}")
