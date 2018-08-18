emu.speedmode("normal") -- Set the speed of the emulator

frameNumber = 0;

--load savestate
--https://stackoverflow.com/questions/5303174/how-to-get-list-of-directories-in-lua
function dirLookup(dir)
	files = {}
   local p = io.popen('find "'..dir..'" -type f')  --Open directory look for files, save data in p. By giving '-type f' as parameter, it returns all files.     
   for file in p:lines() do                         --Loop through all files
	   table.insert(files,file)
   end
   return files;
end

filename =nil
local files =dirLookup("/opt/train/stateSaving/saveStates/");

for i=1, #files, 1 do
--   for f in paths.files("/opt/train/stateSaving/saveStates/") do
                if files[i]:match("test.fcs$") then --level-distance.fcs aka number-number.fcs
                        filename =files[i]
                        gui.text(5,50, "" ..filename);
                        emu.pause(); --make it obvious there is an error
                end;

   end
saveObject = savestate.create(filename);

--saveObject = savestate.create(1);
 savestate.load(saveObject);


while true do
  if (emu.emulating()) then
     -- Execute instructions for FCEUX
     frameNumber = frameNumber + 1;
     if (frameNumber % 400 == 0) then
        savestate.save(saveObject);
        savestate.persist(saveObject);	

		--lets copy that file, but rename it according to frameNumber
		infile = io.open(filename,"rb");
		source_content = infile:read("*all")
		new_saved_state_file = "/opt/train/stateSaving/saveStates/" .. frameNumber .. ".fcs"
		file = io.open(new_saved_state_file, "wb")
		file:write(source_content)
		file:close();

	 if (frameNumber % 400 < 10) then gui.text(50,50, "Saved"); end; --give it some time to show
     end;
     emu.frameadvance() -- This essentially tells FCEUX to keep running
     --end if player is dead (copied from get_is_dead)
  end;
  local player_state = memory.readbyte(0x000e);
  if (player_state == 0x06) or (player_state == 0x0b) then 
	 emu.softreset();
     savestate.load(saveObject);
--break 
   end;   
end
