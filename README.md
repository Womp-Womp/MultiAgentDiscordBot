# Discord AI Bot

This Discord bot leverages advanced AI models and custom functionalities to interact with users, process commands, and provide a rich, character-driven experience. It integrates services like OpenAI's GPT models, web scraping, and a unique character system to engage users in a dynamic and interactive manner.

## Features

- **Character System**: Users can select from a variety of characters, each with its own unique personality and response style, enhancing the interaction experience.
- **Advanced AI Integration**: Utilizes OpenAI's GPT models for generating responses, ensuring high-quality and contextually relevant interactions.
- **Custom Commands**: Supports a range of commands, including web searches, user interactions, and administrative functions like balance management.
- **Dynamic Interaction**: Through a supervisor system, the bot intelligently routes tasks between different agents (characters) based on the user's input and context.
- **Token System**: Incorporates a token-based system for tracking and managing user interactions, with support for balance checks and top-ups.

## Prerequisites

- Python 3.8 or higher
- Discord.py library
- OpenAI API key
- SQLite3 for database management
- Additional Python packages: `requests`, `beautifulsoup4`, `pydantic`, `transformers`, `glob`, `aiohttp`, `json`, `duckduckgo_search`

## Setup

1. **Clone the Repository**:
"git clone "

2. **Install Dependencies**:
"pip install -r requirements.txt"

3. **Configure Environment Variables**:
- `DISCORD_BOT_TOKEN`: Your Discord bot token.
- `OPENAI_API_KEY`: Your OpenAI API key for accessing GPT models.

4. Database config:
TODO

## Usage

- `/select_character <character_name>`: Selects a character for personalized interactions.
- `/interact <query>`: Engages with the bot using the selected character's persona.
- `/balance`: Checks the user's current token balance.
- `/buy-more-tokens`: Provides information on how to top up tokens.
- `/help`: Displays information on how to use the bot and its features.

## Customization

- **Adding New Characters**: To introduce new characters, extend the `character_prompts` dictionary with the character's name as the key and their corresponding personality prompt as the value.
- **Modifying Commands**: Commands can be added or altered in the `bot.py` file using the `@bot.slash_command` decorator.

## Support

For support, questions, or contributions, please open an issue or pull request in the GitHub repository.

## License

This project is licensed under the [MIT License](LICENSE).


