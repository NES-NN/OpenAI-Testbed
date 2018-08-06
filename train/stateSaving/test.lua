emu.speedmode("normal") -- Set the speed of the emulator

frameNumber = 0;

--load savestate
saveObject = savestate.create("/opt/train/stateSaving/saveStates/test.fcs");
--saveObject = savestate.create(1);
 savestate.load(saveObject);


while true do
  if (emu.emulating()) then
     -- Execute instructions for FCEUX
     frameNumber = frameNumber + 1;
     if (frameNumber % 400 == 0) then
        savestate.save(saveObject);
        savestate.persist(saveObject);	
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
