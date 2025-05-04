let tweetsCollected = new Set();
let scrollInterval;

function startCollectingTweets() {
  scrollInterval = setInterval(() => {
    window.scrollBy(0, 1500); // Auto-scroll down

    document.querySelectorAll('article[data-testid="tweet"]').forEach(article => {
      const userLink = article.querySelector('a[href*="/status/"]');
      const timeTag = article.querySelector('time');
      const username = userLink ? userLink.href.split("/")[3] : null;
      const date = timeTag ? timeTag.getAttribute("datetime") : null;
      const text = article.innerText.replace(/\s+/g, ' ').trim();

      if (username && date && text) {
        tweetsCollected.add(`${username} | ${date}\n${text}`);
      }
    });

    console.log(`Collected ${tweetsCollected.size} tweets...`);
  }, 1500);
}

function stopAndDownload() {
  clearInterval(scrollInterval);
  const blob = new Blob([Array.from(tweetsCollected).join('\n\n')], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "tweets_with_date.txt";
  a.click();
  URL.revokeObjectURL(url);
  console.log("✅ Download started.");
}

// Step 1: Run to start collecting
startCollectingTweets();

// Step 2: Run to stop and download
/// stopAndDownload();
