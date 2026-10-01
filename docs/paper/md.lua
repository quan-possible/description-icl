-- Pandoc filter for to_md.sh: makes paper.md read like the PDF on GitHub.
-- Adds the title, author and abstract; numbers sections 1, 2, ... and the
-- appendix A, B, ...; turns figure and table captions into plain paragraphs
-- (so their math renders); drops LaTeX-only spans and link attributes; and
-- heads the bibliography.

local function is_app(h) return h.identifier:match("^app:") ~= nil end

function Pandoc(doc)
  local nmain = 0
  for _, b in ipairs(doc.blocks) do
    if b.t == "Header" and b.level == 1 and not is_app(b) then nmain = nmain + 1 end
  end

  local sec, app, fig, tab = 0, 0, 0, 0
  doc = doc:walk {
    Span = function(s) if s.identifier == "" then return s.content end end,
    -- An italic proposition holding a display equation would put the closing
    -- `*` after the math fence and break the block, so keep those upright.
    Emph = function(e)
      local display = false
      e:walk { Math = function(m) if m.mathtype == "DisplayMath" then display = true end end }
      if display then return e.content end
    end,
    Link = function(l)
      local ref = l.attributes["reference-type"]
      l.attributes = {}
      if ref and l.target:match("^#app:") then
        local a, rest = pandoc.utils.stringify(l.content):match("^(%d+)(.*)$")
        if a then l.content = { pandoc.Str(string.char(64 + tonumber(a) - nmain) .. rest) } end
      end
      return l
    end,
  }
  doc = doc:walk {
    Header = function(h)
      if h.level == 1 then
        local label
        if is_app(h) then app = app + 1; label = string.char(64 + app)
        else sec = sec + 1; label = tostring(sec) end
        h.content:insert(1, pandoc.Space()); h.content:insert(1, pandoc.Str(label))
      end
      if h.level < 4 then h.level = h.level + 1 end  -- the title takes level 1; run-in paragraphs stay at 4
      return h
    end,
    Figure = function(f)
      fig = fig + 1
      local img
      f.content:walk { Image = function(i) img = i end }
      img.caption = {}
      local cap = pandoc.utils.blocks_to_inlines(f.caption.long)
      cap:insert(1, pandoc.Space()); cap:insert(1, pandoc.Strong("Figure " .. fig .. "."))
      return { pandoc.Para { img }, pandoc.Para(cap) }
    end,
    Table = function(t)
      tab = tab + 1
      local cap = pandoc.utils.blocks_to_inlines(t.caption.long)
      t.caption.long = {}
      cap:insert(1, pandoc.Space()); cap:insert(1, pandoc.Strong("Table " .. tab .. "."))
      return { t, pandoc.Para(cap) }
    end,
  }

  local m, front = doc.meta, pandoc.Blocks {}
  front:insert(pandoc.Header(1, m.title))
  front:insert(pandoc.Para(m.author[1]))
  if m.date then front:insert(pandoc.Para(pandoc.Emph(m.date))) end
  front:insert(pandoc.Header(2, "Abstract"))
  front:extend(m.abstract)
  doc.blocks = front .. doc.blocks:walk {
    Div = function(d) if d.identifier == "refs" then return { pandoc.Header(2, "References"), d } end end,
  }
  return doc
end
