# Deploying LazyMeets on Streamlit Community Cloud

This sets up the public site with "Sign in with Google", the free preloaded Groq key, and the 60-minutes-a-day limit.

## 1. Push the code to GitHub

Push everything in this folder. **Don't** push `.env` or `.streamlit/secrets.toml` (both are in `.gitignore`).

## 2. Create the Google sign-in (about 10 minutes)

1. Go to [console.cloud.google.com](https://console.cloud.google.com) and create a new project (any name, e.g. "LazyMeets").
2. Open **APIs & Services → OAuth consent screen** (it may be called **Google Auth Platform**) and click **Get started**:
   - App name: `LazyMeets`, support email: yours
   - Audience: **External**
   - Contact email: yours, then create.
3. Under **Audience**, click **Publish app**. Otherwise only the test users you list can sign in. The app only asks for name and email, so Google doesn't need to review it.
4. Open **Clients → Create client**:
   - Application type: **Web application**
   - Authorized redirect URIs, add both:
     - `https://YOUR-APP.streamlit.app/oauth2callback` (your real app URL)
     - `http://localhost:8501/oauth2callback` (for testing on your own computer)
   - Create, then copy the **Client ID** and **Client secret**.

## 3. Deploy on Streamlit

1. On [share.streamlit.io](https://share.streamlit.io), click **Create app → Deploy a public app from GitHub**.
2. Repository: yours. Branch: `main`. Main file path: `app.py` (or `lazymeets/app.py` if the code is inside a folder).
3. Open **Advanced settings**:
   - Python version: **3.11** or **3.12**
   - **Secrets**: paste the contents of `.streamlit/secrets.toml.example` and fill in:
     - your Groq key
     - `redirect_uri`: your app URL + `/oauth2callback`
     - `cookie_secret`: any long random string
     - the Google client ID and secret
4. Click **Deploy**. The first build takes a few minutes.

## 4. Check it

- Open the app → you should see **Sign in with Google**.
- Sign in → choose **free key** or **your own keys**.
- With the free key, the sidebar shows how many minutes are left today.

## Good to know

- **People sign in every time they open the site.** A remembered sign-in from an earlier visit isn't accepted: the app signs them out and shows the sign-in page again. Google's "choose an account" screen appears each time because of `client_kwargs = { prompt = "select_account" }` in the secrets.
- **Sign out** is at the top of the setup pages and in the sidebar of the app. It also forgets any API keys pasted in that tab.
- **The daily allowance** is counted per Google account, in audio minutes, and resets at midnight India time (set `QUOTA_TZ` to change it). It's stored in a small file on the server. Streamlit Cloud wipes files when the app restarts or goes to sleep, and that also resets the counts. That's fine for a daily limit.
- **People who bring their own keys** aren't limited by the app. Their keys stay in their browser session and are never saved.
- **Sign-in "redirect_uri_mismatch" error** means the URI in Google Console doesn't exactly match `redirect_uri` in your secrets (check https vs http and the trailing `/oauth2callback`).
- **To run without sign-in** (for example on your own computer), leave out the `[auth]` block. The app then skips the sign-in page.
