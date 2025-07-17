# AI-Fashion-Wardrobe
AI Fashion Stylist is a web app that lets users upload clothing images, run an AI script to generate weekly outfits, and view results in a sleek UI. It features image preview/removal, real-time terminal logs, and auto-organized daily outfit displays.


---

# 👗 AI Fashion Stylist

**AI Fashion Stylist** is an AI-powered web application that helps users generate personalized outfit suggestions using images from their own wardrobe. Designed with a modern, minimalist fashion aesthetic, the app allows users to upload clothing images, process them through an AI engine, and receive a full week’s worth of stylish outfit combinations.

The application features an intuitive image upload interface where users can preview selected images and remove any by clicking a cross (×) icon. Once finalized, users can click the **Upload to Wardrobe** button, which clears previous entries and uploads new clothing images to a dedicated folder called `Fashion Clothes`.

Clicking **Generate Outfits** runs the core AI script (`main.py`) that processes the uploaded clothing items to create combinations. During this process, the app displays a sleek loading screen that streams real-time terminal output from the backend, giving users a transparent view of the AI’s progress.

After processing, the app loads outfit suggestions organized into folders named `Day-1` to `Day-7` inside a directory called `Final_Combinations`. These outfits are then displayed on the website under the **Daily Outfits** section, grouped by day in a visually engaging and easy-to-browse layout.

### Tech Stack:
- **Frontend**: React with Tailwind CSS for clean, responsive design
- **Backend**: Python with Flask or FastAPI for file handling and script execution
- **AI Engine**: Python-based logic in `main.py` for outfit generation

### Key Features:
- Upload and preview clothing images
- Remove individual images before uploading
- Real-time logging of backend AI process
- Automatically displays final outfit results in daily groups
- Elegant, mobile-friendly UI inspired by fashion magazines

**AI Fashion Stylist** is perfect for anyone looking to automate their outfit planning with the power of AI and style. Whether you're prepping for the week or just need daily inspiration, this app brings personalized fashion right to your screen.


