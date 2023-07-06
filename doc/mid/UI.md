# UI

UI system provide a GUI for user to interact with all other systems. It may include multiple applications, online or offline. This UI system refer to an application running locally on users machine.


## Race UI

Main function is to visualize a race. Key functions are
- Setup
- Run
- Playback

When render a race, UI should display time elapsed, each car's progress. 

May allow zoom to see more details if the whole track is to large, provide enough resolution to show the area of interest.

### Offline mode

In offline mode, UI can allow user to provide all required data to setup a race. Once started, Race system will generate all race trace data, and same them into a folder. 

Then UI can use saved race trace data to render a race. UI can support pause, step by step, fast forward, going back etc. 

### Online mode

In online mode, UI can update the UI content with race progress in realtime.

Considering possible network jitter, lost of data, or Race system failure, Online mode UI may have a delay function. It can buffer data, render UI at 5 seconds delay of real time, may retry send data and get progress to online Race system.


### Interactive mode
Allow user to interactive control a car instead of using AI model to drive it, and record race data. 
This could help user to understand how the system work with user act as the AI model. 

Possible allow a user driven car race with other AI model driven cars. 


