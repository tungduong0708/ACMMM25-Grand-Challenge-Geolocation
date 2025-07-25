import asyncio
import json
import logging
from pathlib import Path
from typing import Literal, TypedDict

from playwright.async_api import Page, async_playwright

READABILITY_JS_URL = "https://unpkg.com/@mozilla/readability@0.4.4/Readability.js"
logger = logging.getLogger("uvicorn.error")


class PageText(TypedDict):
    url: str
    text: str


WaitUntil = Literal["load", "domcontentloaded", "networkidle", "commit"]


async def _inject_readability(page: Page) -> None:
    await page.add_script_tag(url=READABILITY_JS_URL)
    await page.add_script_tag(
        content="window.__readability__ = new Readability(document.cloneNode(true));"
    )


async def _fetch_text(page: Page, url: str, wait_until: WaitUntil) -> str:
    await page.goto(url, wait_until=wait_until)
    await page.wait_for_timeout(1000)

    # Attempt Readability.js parsing first
    try:
        await _inject_readability(page)
        readability_text = await page.evaluate(
            "() => window.__readability__.parse()?.textContent"
        )
        if readability_text:
            return readability_text.strip()
    except BaseException as _:
        pass

    # Fallback: Twitter specific logic
    try:
        tweet_text = await page.locator(
            "article div[data-testid='tweetText']"
        ).all_inner_texts()
        if tweet_text:
            return "\n".join(tweet_text)
    except BaseException as _:
        pass

    # Final fallback: full body text
    return await page.evaluate("() => document.body.innerText")


async def fetch_text(
    url: str, headless: bool = False, wait_until: WaitUntil = "load"
) -> PageText:
    async with async_playwright() as pw:
        browser = await pw.chromium.launch_persistent_context(
            user_data_dir="",
            channel="chrome",
            headless=headless,
            no_viewport=True,
        )
        page = await browser.new_page()
        text = await _fetch_text(page, url, wait_until)
        await browser.close()

    return PageText(url=url, text=text)


async def fetch_texts(
    urls: list[str], headless: bool = False, wait_until: WaitUntil = "load"
) -> list[PageText | BaseException]:
    async with async_playwright() as pw:
        browser = await pw.chromium.launch_persistent_context(
            user_data_dir="",
            channel="chrome",
            headless=headless,
            no_viewport=True,
        )
        pages = [await browser.new_page() for _ in urls]

        tasks = [_fetch_text(page, url, wait_until) for page, url in zip(pages, urls)]
        results_raw = await asyncio.gather(*tasks, return_exceptions=True)
        await browser.close()

    results: list[PageText | BaseException] = []
    for url, result in zip(urls, results_raw):
        if isinstance(result, BaseException):
            results.append(result)
        else:
            results.append(PageText(url=url, text=result))

    return results


async def fetch_links_to_json(
    links: list[str],
    output_path: str,
    headless: bool = False,
    wait_until: WaitUntil = "load",
    max_content_length: int = 5000,
) -> None:
    """
    Fetch content from a list of links and save to a JSON file.

    Args:
        links: List of URLs to fetch content from
        output_path: Path where the JSON file will be saved
        headless: Whether to run browser in headless mode
        wait_until: When to consider page loading complete
        max_content_length: Maximum number of characters to keep from each page content

    Returns:
        None (saves results to JSON file)
    """
    logger.info(f"📥 Fetching content from {len(links)} links...")

    # Fetch content from all links
    results = await fetch_texts(links, headless=headless, wait_until=wait_until)

    # Process results into the desired format
    json_data = []
    for i, (link, result) in enumerate(zip(links, results)):
        logger.info(f"  Processing {i + 1}/{len(links)}: {link}")

        if isinstance(result, BaseException):
            # Handle errors gracefully
            json_data.append({"link": link, "content": "Fail to fetch content..."})
        else:
            # Successfully fetched content - apply length limit
            content = result["text"]
            if len(content) > max_content_length:
                content = (
                    content[:max_content_length]
                    + "... [content truncated due to length limit]"
                )
                logger.info(
                    f"✂️ Content truncated from {len(result['text'])} to {max_content_length} characters"
                )

            json_data.append({"link": link, "content": content})

    # Ensure output directory exists
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save to JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    logger.info(f"💾 Saved content from {len(links)} links to {output_path}")

    # Print summary
    successful = sum(
        1 for item in json_data if not item["content"].startswith("Error fetching")
    )
    failed = len(json_data) - successful
    logger.info(f"📊 Summary: {successful} successful, {failed} failed")
