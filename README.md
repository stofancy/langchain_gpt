## How to launch it:

0. You need to setup python3 environment with `pip` and `poetry` tool.
0. Create a `.env` file at the root folder.
0. Find a way to get the oPenAI API key, and put in `.env`:
	```
	OPENAI_API_KEY={key of openai}	
	```

0. Launch it:
	``` bash
	poetry install
	poetry run streamlit run ./hackathon.py
	```
0. Generate the meeting transcript, and upload it in webpage. or directly upload `meeting_minutes.md` in project folder.
0. You'll got the meanful summary after a while.

## Next steps

* Integrate with Teachable Machine Image Model for identiting current user.
* Integrate with teams for get meeting transcript automatically.
* integrate with Jira for update the ticket progress, assignee, status, etc,.