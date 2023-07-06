# Competition

Competition system organize an competition events, consist of multile cars and multiple rounds of races. It may use different variation of tracks, setup different combination of cars in each race, and keep race score and each car's ranking. 

Based on each car's performance, it will select cars to advance, organize teams, playoff games, and decide final winners.

## Game

A game is defined a race between 2 teams and a track setup. Each team consist of 3 cars (each with its own model). 

The track setup can be same or different between different games. Track may randomly change 10% center tiles from Road type to Shoulder type. 

At end of each game, each car will get a finish position between 0 and 5, each earn a score = ```5 - position``` . Team score is the sum of 3 cars' position. Team with more score win the game, earn a ranking score 2. 


## Round

Each round will randomly divide cars in teams, and randomly pairing teams to race. All games in same round use the same track track. Each car will earn the ranking point its team get in that round.

There are 12 qualification rounds. At end of qualification rounds, teams are ranked using their ranking points.

## Team selection

8 teams, each of 4 cars will be organized. Top 8 ranked team will become team caption. Car seletion in the order of: 1 to 8, 8 to 1, 1 to 8.

## Playoff

Each playoff level (quarter-final, semi-final, final) has 3 rounds, each rounds use same track. 

Each team will have 3 cars run in a game. Each car have to play at least 1 game in 3 games at each playoff level.

Team won 2 games will advance to next level. 

Champion team is the winner of final level.



