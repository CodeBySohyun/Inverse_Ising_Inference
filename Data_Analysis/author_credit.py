from markupsafe import escape
from bokeh.models import Div

def add_author_table(name: str, website_url: str, icon_url: str) -> Div:
    """
    Add author credit to a table with name, website URL, and icon URL.

    Args:
        name (str): The name of the author.
        website_url (str): The URL of the author's website.
        icon_url (str): The URL of the author's icon.

    Returns:
        Div: The author credit table.
    """
    if not name or not website_url or not icon_url:
        raise ValueError("Name, website URL, and icon URL cannot be empty.")
    if not isinstance(name, str) or not isinstance(website_url, str) or not isinstance(icon_url, str):
        raise TypeError("Name, website URL, and icon URL must be strings.")

    name = escape(name)
    website_url = escape(website_url)
    icon_url = escape(icon_url)

    author_table = Div(text=f"""
    <table style="border: 0; padding: 0;">
        <tr>
            <td style="padding: 0;">
                <a href="{website_url}" target="_blank">
                    <img src="{icon_url}" style="height: 14pt; width: 14pt; vertical-align: middle;">
                </a>
            </td>
            <td style="padding: 0; vertical-align: middle;">
                <span style="font-size: 12pt;"><b>&nbsp;{name}</b></span>
            </td>
        </tr>
    </table>
    """)

    return author_table