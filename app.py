from src.langgraphagenticai.main import load_langgraph_agenticai_app

if __name__=="__main__":
    load_langgraph_agenticai_app()

    #we are calling the load_langgraph_agenticai_app function to start the application. This function is responsible for initializing and running the LangGraph AgenticAI application, which may include setting up necessary configurations, loading models, and starting any required services or servers.
    # app.py will call  main.py -> load_langgraph_agenticai_app
    # main.py -> load_langgraph_agenticai_app -> calling ##Load UI ->from loadui.py
    #loadui.py will call uiconfigfile.py to get the config file and load the UI with the options from the config file.
    #