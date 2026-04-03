from huggingface_hub import HfApi, login, create_repo, add_space_secret

TOKEN = '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'
WEBHOOK = 'https://discord.com/api/webhooks/1489653442455535758/1FklLHpPcV8c-Wp4-6uNIelFVaB1SwPgqa6uKa7WK-s3wJIpVFiQQ2OgFLsB_ytwzMaq'
REPO_ID = 'Alaqmar/WasItAI-backend'

def main():
    print(f"🚀 Starting Cloud Backend Setup for {REPO_ID}...")
    login(TOKEN)
    api = HfApi()

    # 1. Create Repository
    try:
        print("🛠️ Creating Space repository...")
        api.create_repo(
            repo_id=REPO_ID, 
            repo_type='space', 
            space_sdk='docker', 
            private=True
        )
        print("✅ Space created successfully!")
    except Exception as e:
        if "already exists" in str(e).lower():
            print("ℹ️ Space already exists, continuing...")
        else:
            print(f"⚠️ Error creating space: {e}")
            return

    # 2. Add Discord Secret
    try:
        print("🔗 Adding Discord Webhook secret...")
        api.add_space_secret(
            repo_id=REPO_ID, 
            key='DISCORD_WEBHOOK_URL', 
            value=WEBHOOK
        )
        print("✅ Secret added!")
    except Exception as e:
        print(f"⚠️ Error adding secret: {e}")

    # 3. Final URL Check
    print(f"\n🌍 YOUR LIVE BACKEND URL WILL BE:")
    print(f"https://alaqmar-wasitai-backend.hf.space")
    print(f"\n(It may take 5-10 minutes for the first build to finish)")

if __name__ == "__main__":
    main()
