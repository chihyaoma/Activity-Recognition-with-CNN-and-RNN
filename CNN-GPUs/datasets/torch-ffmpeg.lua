do
  local FFmpeg = torch.class('FFmpeg')

  function FFmpeg:__init(video_path, opts)
    self.video_path = video_path
    self.opts = opts or ''
    self.valid = true
    self.fd = nil
  end

  function FFmpeg:read(nframes)
    if self.fd == nil then
      -- open ffmpeg pipe
      -- this subprocess will send raw RGB values to us, corresponding to frames
      local cmd = 'ffmpeg -i ' .. self.video_path .. ' ' .. self.opts .. ' -f image2pipe -pix_fmt rgb24 -loglevel fatal -vcodec ppm -'
      self.fd = assert(torch.PipeFile(cmd))
      self.fd:binary()
      self.fd:quiet()
    end

    -- read nframes from the pipe
    local t
    local t2
    local dim = {}

    for i=1,nframes do
      local magic_str = self.fd:readString("*l")
      local dim_str = self.fd:readString("*l")
      local max_str = self.fd:readString("*l")

      if self.fd:hasError() then
        self.valid = false
        return nil
      end

      assert(magic_str == "P6")
      assert(tonumber(max_str) == 255) 

      if i == 1 then 
        for k in string.gmatch(dim_str, '%d+') do table.insert(dim, tonumber(k)) end
        assert(#dim == 2)

        t = torch.ByteTensor(nframes, dim[2], dim[1], 3):fill(0)
        t2 = torch.ByteTensor(dim[2], dim[1], 3)
      end

      self.fd:readByte(t2:storage())
      t[i]:copy(t2)

      if self.fd:hasError() then
        self.valid = false
        return nil
      end
    end

    return t:permute(1,4,2,3)
  end

  function FFmpeg:close()
    if self.fd ~= nil then
      self.fd:close()
      self.fd = nil
      self.valid = false
    end
  end

  function FFmpeg:stats() 
    -- use ffprobe to find width/height of video
    -- this will store self.width, self.height, self.duration
    local cmd = 'ffprobe -select_streams v -v error -show_entries stream=width,height,duration -of default=noprint_wrappers=1 ' .. self.video_path
    local fd = assert(torch.PipeFile(cmd))
    fd:quiet()

    local retval = {}

    for i=1,3 do
      local line = fd:readString('*l')
      if fd:hasError() then
        self.valid = false
        break
      end
      local split = {}
      for k in string.gmatch(line, '[^=]*') do table.insert(split, k) end
      retval[split[1]] = tonumber(split[3])
    end

    fd:close()

    return retval
  end
end
