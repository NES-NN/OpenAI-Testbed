emu.speedmode("normal") -- Set the speed of the emulator

-- ===========================
--         Constants
-- ===========================
level_matrix = {
    { 1, 1, 1 }, { 1, 2, 3 }, { 1, 3, 4 }, { 1, 4, 5 },
    { 2, 1, 1 }, { 2, 2, 3 }, { 2, 3, 4 }, { 2, 4, 5 },
    { 3, 1, 1 }, { 3, 2, 2 }, { 3, 3, 3 }, { 3, 4, 4 },
    { 4, 1, 1 }, { 4, 2, 3 }, { 4, 3, 4 }, { 4, 4, 5 },
    { 5, 1, 1 }, { 5, 2, 2 }, { 5, 3, 3 }, { 5, 4, 4 },
    { 6, 1, 1 }, { 6, 2, 2 }, { 6, 3, 3 }, { 6, 4, 4 },
    { 7, 1, 1 }, { 7, 2, 3 }, { 7, 3, 4 }, { 7, 4, 5 },
    { 8, 1, 1 }, { 8, 2, 2 }, { 8, 3, 3 }, { 8, 4, 4 }
};
start_delay = 175; -- Delay before pressing the start button
save_directory = "/opt/train/stateSaving/saveStates-josh"

-- ===========================
--         Variables
-- ===========================
loaded_and_saved_state = 0  -- Indicates whether we have managed to successfully load and save state for a level
is_hook_set = 0;            -- Indicates whether the current world, level and area hooks have been set
is_started = 0;             -- Indicates that the timer has started to decrease (i.e. commands can now be processed)
is_finished = 0;            -- Indicates a life has been lost, world has changed, or finish line crossed
last_time_left = 0;         -- Indicates the last time left (to check if timer has started to decrease)
force_refresh = 0;          -- Forces to return full screen (all pixels and data) for this number of frames
commands = {};              -- List of current commands (inputs)
running_thread = 0;         -- To avoid 2 threads running at the same time
target_world = nil;         -- The target world
target_level = nil;         -- The target level
target_area = nil;          -- The target area
filename = nil;             -- The filename to save state out to

--resets all variables back to defaults
function reset_vars()
    loaded_and_saved_state = 0;
    is_hook_set = 0;
    is_started = 0;
    is_finished = 0;
    last_time_left = 0;
    force_refresh = 0;
    commands = {};
    running_thread = 0;
    target_world = nil;
    target_level = nil;
    target_area = nil;
    filename = nil;
end;

-- ===========================
--         Memory Addresses
-- ===========================
addr_world = 0x075f;
addr_level = 0x075c;
addr_area = 0x0760;
addr_time = 0x07f8;

-- ===========================
--         Save State
-- ===========================
function save_state_to_file()
    savestate_object = savestate.create(save_directory .. "/" .. target_world .. "-" .. target_level .. "-" .. target_area .. ".fcs")
    savestate.save(savestate_object);
    savestate.persist(savestate_object);  
end;

-- ===========================
--         Functions
-- ===========================
function hook_set_world()
    memory.writebyte(addr_world, (target_world - 1));
    memory.writebyte(addr_level, (target_level - 1));
    memory.writebyte(addr_area, (target_area - 1));
end;

function hook_set_level()
    memory.writebyte(addr_world, (target_world - 1));
    memory.writebyte(addr_level, (target_level - 1));
    memory.writebyte(addr_area, (target_area - 1));
end;

function hook_set_area()
    memory.writebyte(addr_world, (target_world - 1));
    memory.writebyte(addr_level, (target_level - 1));
    memory.writebyte(addr_area, (target_area - 1));
end;

-- readbyterange - Reads a range of bytes and return a number
function readbyterange(address, length)
  local return_value = 0;
  for offset = 0,length-1 do
    return_value = return_value * 10;
    return_value = return_value + memory.readbyte(address + offset);
  end;
  return return_value;
end

-- get_time - Returns the time left (0 to 999)
function get_time()
    return tonumber(readbyterange(addr_time, 3));
end;

-- check_if_started - Checks if the timer has started to decrease
-- this is to avoid receiving commands while the level is loading, or the animation is still running
function check_if_started()
    local time_left = get_time();

    -- Cannot start before 'start' is pressed
    local framecount = emu.framecount();
    if (framecount < start_delay) then
        return;
    end;

    -- Checking if time has decreased
    if (time_left > 0) and (is_finished ~= 1) then
        -- Level started (if timer decreased)
        if (last_time_left > time_left) then
            is_started = 1;
            last_time_left = 0;
            force_refresh = 5;  -- Sending full screen for next 5 frames, then only diffs
        else
            last_time_left = time_left;
        end;
    end;
    return;
end;

-- load a particular level and save its state out to a file
function load_and_save_state()
    if running_thread == 1 then
        return;
    end;
    running_thread = 1;

    --load the level hook registers
    if 0 == is_hook_set then
        memory.registerwrite(addr_world, hook_set_world);
        memory.registerwrite(addr_level, hook_set_level);
        memory.registerwrite(addr_area, hook_set_area);
        is_hook_set = 1;
    end;

    --load state likely messes with framecount, so moving below
    local framecount = emu.framecount();

    -- Checking if game is started
    if is_started == 0 then
        check_if_started();

    -- Game has started; save state and move to next
    elseif is_started == 1 then
        -- Let the game run for a bit before saving
        for i=1,50,1 do
            emu.frameadvance();
        end

        -- Save the level state to file
        save_state_to_file();

        -- Signal that we have finished
        loaded_and_saved_state = 1;
        return;
    end;

    -- Checking if game has started, if not, pressing "start" to start it
    if (0 == is_started) and (framecount == start_delay) then
        commands["start"] = true;
        joypad.set(1, commands);
        emu.frameadvance();
        commands["start"] = false;

    -- Game not yet started, just skipping frame
    elseif 0 == is_started then
        emu.frameadvance();
    end;

    force_refresh = force_refresh - 1;
    if force_refresh < 0 then force_refresh = 0; end;
    running_thread = 0;
end;

-- ===========================
--         Main function
-- ===========================
for i=1, #level_matrix, 1 do
    -- Reset local state
    reset_vars()

    -- Load up memory registers
    target_world = level_matrix[i][1];
    target_level = level_matrix[i][2];
    target_area = level_matrix[i][3];

    -- Load and save the level state
    while 0 == loaded_and_saved_state do
        load_and_save_state()
    end;

    -- Execute a reset
    emu.softreset()
end;
os.exit();
