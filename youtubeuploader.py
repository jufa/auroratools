from pathlib import Path
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle

SCOPES = ["https://www.googleapis.com/auth/youtube.upload",
          "https://www.googleapis.com/auth/youtube"]


# Path to token file
token_file = Path("./secrets") / "token.pickle"
google_secret = Path("./secrets") / "client_secret_google.json"


# Load saved credentials if they exist
def get_youtube_client():
  if token_file.exists():
      with token_file.open("rb") as f:
          creds = pickle.load(f)

  # If no valid credentials, run OAuth flow
  if not creds or not creds.valid:
      if creds and creds.expired and creds.refresh_token:
          creds.refresh(Request())
      else:
          flow = InstalledAppFlow.from_client_secrets_file(
              google_secret,
              SCOPES
          )
          creds = flow.run_local_server(port=8080)
      
      # Save credentials for next time
      with token_file.open("wb") as f:
          pickle.dump(creds, f)

  return build("youtube", "v3", credentials=creds)


def upload_video(youtube, file_path, title, description, playlist_id=None):
    body = {
        "snippet": {
            "title": title,
            "description": description,
            "categoryId": "28",   # Science & Technology
        },
        "status": {
            "privacyStatus": "public"
        }
    }

    media = MediaFileUpload(file_path, chunksize=-1, resumable=True)

    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media
    )

    response = request.execute()
    video_id = response["id"]
    print("Uploaded:", video_id)

    # Add to playlist (optional)
    if playlist_id:
        youtube.playlistItems().insert(
            part="snippet",
            body={
                "snippet": {
                    "playlistId": playlist_id,
                    "resourceId": {
                        "kind": "youtube#video",
                        "videoId": video_id
                    }
                }
            }
        ).execute()
        print("Added to playlist:", playlist_id)

    return video_id
