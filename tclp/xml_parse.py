# xml_parse.py
"""
This script is used to parse the XML file downloaded from the WordPress site and extract the content of the clauses, guides, and glossary terms.

Users will likely not need to run this script as the information has already been extracted and cleaned for you.

However, this file is included in the repository for transparency and to show how the data was extracted and cleaned.

And for those that might wish to start from scratch."""

#  Necessary imports
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import os
import re

# setting the stage with necessary variables and namespaces

# NOTE: The XML file must be downloaded from the WordPress site and saved in the same directory as this script
# Please name it "thechancerylaneproject.WordPress.xml" or update the file name in the code below
tree = ET.parse("data/thechancerylaneproject.WordPress.xml")

root = tree.getroot()
content_namespace = "{http://purl.org/rss/1.0/modules/content/}"
wp_namespace = "{http://wordpress.org/export/1.2/}"

# make a new directory to store the text files
output_dir = "data/tclp_raw_content"
os.makedirs(output_dir, exist_ok=True)

# parsing the XML file and extracting the content; this code will extract clauses, guides, and glossary terms
target_post_types = {"clause", "guide", "glossary-term"}
counter = 0

for item in root.findall("channel/item"):
    # find the post type for each item
    post_type_element = item.find(f"{wp_namespace}post_type")
    post_type = post_type_element.text if post_type_element is not None else ""

    # check that content is one of the desired post types
    if post_type in target_post_types:
        title = (
            item.find("title").text if item.find("title") is not None else "No Title"
        )

        cleaned_content = ""

        # glossary terms are a special case, as they have multiple definitions so we need to extract them all
        # this means this code is the most "hard coded" and if ever content about glossary terms changes, this code will need to be updated

        # I am changing this so that it excludes drafting notes
        if post_type == "glossary-term":
            definitions = []
            for meta in item.findall(
                "wp:postmeta", namespaces={"wp": "http://wordpress.org/export/1.2/"}
            ):
                meta_key = meta.find(f"{wp_namespace}meta_key").text
                meta_value = (
                    meta.find(f"{wp_namespace}meta_value").text
                    if meta.find(f"{wp_namespace}meta_value") is not None
                    else ""
                )

                # Currently, glossary content is stored under "drafting_notes" and "term_definition_" meta keys
                if meta_key.startswith("term_definition_"):
                    soup = BeautifulSoup(meta_value, "html.parser")
                    definitions.append(soup.get_text(separator="\n").strip())
            cleaned_content = "\n\n".join(definitions)

        elif post_type == "clause":
            content_element = item.find(f"{content_namespace}encoded")
            full_content = ""

            if content_element is not None and content_element.text:
                soup = BeautifulSoup(content_element.text, "html.parser")
                full_content = soup.get_text(separator="\n").strip()

            # Extract content from `clause_recitals` and add it to `full_content`; this is only necessary for Roberto's Recitals at the moment
            for meta in item.findall(
                "wp:postmeta", namespaces={"wp": "http://wordpress.org/export/1.2/"}
            ):
                meta_key = (
                    meta.find(f"{wp_namespace}meta_key").text
                    if meta.find(f"{wp_namespace}meta_key") is not None
                    else ""
                )
                if meta_key == "clause_recitals":
                    recitals_text = (
                        meta.find(f"{wp_namespace}meta_value").text
                        if meta.find(f"{wp_namespace}meta_value") is not None
                        else ""
                    )

                    # Only parse and add recitals if `recitals_text` is not None or empty
                    if recitals_text:
                        recitals_soup = BeautifulSoup(recitals_text, "html.parser")
                        cleaned_recitals = recitals_soup.get_text(
                            separator="\n"
                        ).strip()
                        full_content = (
                            f"{cleaned_recitals}\n\n{full_content}"
                            if cleaned_recitals
                            else full_content
                        )

                    # Possible start points in order of priority
                    possible_start_points = [
                        "(A)",
                        "1. ",
                        "1",
                        "For",
                        "Sub",
                        "(a)",
                        "General",
                    ]

                    # Initialize start_index to -1 to indicate no start point found initially
                    start_index = -1
                    for start_point in possible_start_points:
                        start_index = full_content.find(start_point)
                        if start_index != -1:
                            # Exit loop once the first start point is found
                            break

            # Extract content from the start point if found
            if start_index != -1:
                cleaned_content = full_content[start_index:].strip()
            else:
                print(f"No valid start point found in content for: {title}")
                cleaned_content = ""

            # Remove all content between "[Drafting note:" and the first "]" after it
            cleaned_content = re.sub(
                r"\[\s*Drafting note:.*?\]",
                "",
                cleaned_content,
                flags=re.DOTALL | re.IGNORECASE,
            )

        else:
            # for other post types, we can just use the main content field
            content_element = item.find(f"{content_namespace}encoded")
            if content_element is not None and content_element.text:
                soup = BeautifulSoup(content_element.text, "html.parser")
                cleaned_content = soup.get_text(separator="\n").strip()

        # Saving all files to the output directory
        if cleaned_content.strip():
            filename = f"{title[:50].replace(' ', '_').replace('/', '_')}.txt"
            filepath = os.path.join(output_dir, filename)

            with open(filepath, "w", encoding="utf-8") as file:
                file.write(cleaned_content)
                counter += 1

            print(f"Saved: {filepath}")
        else:
            print(f"No content to save for {title}")

print(f"Saved {counter} files")
