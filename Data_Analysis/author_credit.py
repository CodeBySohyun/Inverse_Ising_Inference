from bokeh.models import Div

def add_author_credit(name, website_url, icon_url):
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