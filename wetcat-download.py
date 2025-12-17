import os
import sys
import synapseclient
import synapseutils

# -------------------------------------
# 1. SETTINGS — CHANGE THIS ONLY
# -------------------------------------

SSD_NAME = "Extreme SSD"
FOLDER_NAME = "cataract_dataset"  # folder where dataset will be downloaded
SYNAPSE_ID = "syn66401174"  # your dataset ID

# -------------------------------------
# 2. FIND SSD MOUNT POINT
# -------------------------------------

ssd_path = f"/Volumes/{SSD_NAME}"

if not os.path.exists(ssd_path):
    print(f"❌ ERROR: SSD not found at {ssd_path}")
    print("➡️  Make sure your SanDisk SSD is plugged in and check the name in Finder.")
    sys.exit(1)

download_path = os.path.join(ssd_path, FOLDER_NAME)

# Create directory if needed
os.makedirs(download_path, exist_ok=True)

print(f"📁 Download destination: {download_path}")

# -------------------------------------
# 3. LOGIN TO SYNAPSE
# -------------------------------------

syn = synapseclient.Synapse()

try:
    syn.login(authToken="YeyJ0eXAiOiJKV1QiLCJraWQiOiJXN05OOldMSlQ6SjVSSzpMN1RMOlQ3TDc6M1ZYNjpKRU9VOjY0NFI6VTNJWDo1S1oyOjdaQ0s6RlBUSCIsImFsZyI6IlJTMjU2In0.eyJhY2Nlc3MiOnsic2NvcGUiOlsidmlldyIsImRvd25sb2FkIiwibW9kaWZ5Il0sIm9pZGNfY2xhaW1zIjp7fX0sInRva2VuX3R5cGUiOiJQRVJTT05BTF9BQ0NFU1NfVE9LRU4iLCJpc3MiOiJodHRwczovL3JlcG8tcHJvZC5wcm9kLnNhZ2ViYXNlLm9yZy9hdXRoL3YxIiwiYXVkIjoiMCIsIm5iZiI6MTc2NDgyMzQ3MiwiaWF0IjoxNzY0ODIzNDcyLCJqdGkiOiIyOTQzMSIsInN1YiI6IjM1NjY1NDgifQ.txi1Nan_t2yYo4vZctwKZhQQXT7Jr97zc-8xISOgTwMjv6rI73SaOJgdKbevAB1mtO4bA6rR9D7QrbMkewHZS0_OxQMzPqtXoreuaU0uBiLhRBP1bE8zGmsGR590fooiEJiaC56tqtG2Te4atQgXRFhEYPfWwaNDW1bvqu5jvt928tWHmQV_oNvUQS1gNxjkadY0dvw2pCnr_8F00UeSN59nCNCENstZbb1Ey8w_KtpX-hbuv3U9DWrhcqzWNpHTgg56C7MbQuBqjiL89N-9NaNafjkWGad8xzFeJsR-GS-w8n_JMKXQKds7caCcg55dsqyl5m_KOmPUOa465SY4hQ")
    print("🔐 Logged into Synapse successfully.")
except Exception as e:
    print("❌ Synapse login failed:", e)
    sys.exit(1)

# -------------------------------------
# 4. DOWNLOAD DATASET TO THE SSD
# -------------------------------------

print("⬇️ Starting dataset download... This may take a while.")

try:
    files = synapseutils.syncFromSynapse(
        syn,
        SYNAPSE_ID,
        path=download_path,
        ifcollision="keep.local"  # prevents overwriting repeated runs
    )

    print("✅ DOWNLOAD COMPLETE")
    print(f"✔️ Files saved to: {download_path}")

except Exception as e:
    print("❌ Error during download:", e)
    sys.exit(1)

# -------------------------------------
# 5. PRINT DOWNLOADED FILES
# -------------------------------------

print("\n📄 Downloaded files:")
for f in files:
    print(" -", f[path])