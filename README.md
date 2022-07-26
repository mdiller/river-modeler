# River Canyon Heightmap 3D Model Maker

I started with the idea of being able to 3D print a heightmap of the Rogue River canyon. I decided to try to build a more general solution to this problem so that you could apply this to any river. In its current state it is a bit all over the place and not super clean. I may attempt to revisit this and create a webpage out of it for an easier-to-use user experience.

# TODOs

The main todos for this project at the point of the repo's creation
- [x] put all this stuff in a git repo
- [x] flip x direction (i think) (should do this at end of extract_elevations)
- [ ] round edges
- [ ] make the base (prolly my river following dots method)
- [ ] think about doin holes for particular rapids/milestones
  - [ ] could do toothpick-sized holes
  - [ ] manual select for locations
- [ ] make a config file that can have all the settings in it
  - [ ] stuff to define in inches/mm in config

Heres some more todos that are lower priority but would be nice to do after i start the actual printing
- [ ] re-organize the scripts at some point
  - [ ] need a better way of passing the data between parts of the script
- [ ] make it run on a website so other people could use it (python webassembly probably)
  - [ ] make input/search boxes based on osm for selecting river, start/end points, and points of interest


# some links
https://www.openstreetmap.org/way/425061030#map=11/42.5708/-123.8434&layers=Y