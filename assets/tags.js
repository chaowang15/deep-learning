(function () {
  // Homepage filter: by tag (#tag=<slug>) or by month (#month=YYYY-MM).
  // Buttons carry data-filter="tag:<slug>" | "month:YYYY-MM" | "" (show all).
  const cards = Array.from(document.querySelectorAll("[data-note-card]"));
  if (!cards.length) {
    return;
  }

  const list = document.querySelector("[data-note-list]");
  const status = document.querySelector("[data-tag-status]");
  const statusKind = document.querySelector("[data-tag-status-kind]");
  const statusName = document.querySelector("[data-tag-status-name]");
  const statusCount = document.querySelector("[data-tag-status-count]");
  const empty = document.querySelector("[data-tag-empty]");
  const buttons = Array.from(document.querySelectorAll("[data-filter]"));
  const labels = {};
  buttons.forEach((button) => {
    if (button.dataset.filter && button.dataset.filterLabel) {
      labels[button.dataset.filter] = button.dataset.filterLabel;
    }
  });

  let current = "";

  function fromHash() {
    const match = location.hash.match(/^#(tag|month)=([\w-]+)$/);
    const key = match ? `${match[1]}:${match[2]}` : "";
    return labels[key] ? key : "";
  }

  function matches(card, filter) {
    if (!filter) {
      return true;
    }
    const [kind, value] = filter.split(":");
    if (kind === "tag") {
      return card.dataset.tags.split(" ").includes(value);
    }
    return card.dataset.month === value;
  }

  function apply(filter, updateUrl) {
    current = filter;
    let shown = 0;
    cards.forEach((card) => {
      const visible = matches(card, filter);
      card.hidden = !visible;
      if (visible) {
        shown += 1;
        card.classList.remove("is-entering");
        void card.offsetWidth;
        card.classList.add("is-entering");
      }
    });

    buttons.forEach((button) => {
      if (button.hasAttribute("data-filter-clear")) {
        return;
      }
      const on = button.dataset.filter === filter;
      button.classList.toggle("is-active", on);
      button.setAttribute("aria-pressed", String(on));
      const row = button.closest(".tag-filter--row");
      if (on && row && row.scrollWidth > row.clientWidth) {
        row.scrollLeft = button.offsetLeft - row.offsetLeft - 18;
      }
    });

    if (status) {
      status.hidden = !filter;
      if (filter) {
        statusKind.textContent = filter.startsWith("tag:") ? "标签" : "月份";
        statusName.textContent = labels[filter] || filter;
        statusName.classList.toggle("is-tag", filter.startsWith("tag:"));
        statusCount.textContent = shown;
      }
    }
    if (empty) {
      empty.hidden = shown > 0;
    }

    if (updateUrl) {
      const url = filter ? `#${filter.replace(":", "=")}` : location.pathname + location.search;
      history.replaceState(null, "", url);
    }
  }

  document.addEventListener("click", (event) => {
    const button = event.target.closest("[data-filter]");
    if (!button) {
      return;
    }
    event.preventDefault();
    const filter = button.dataset.filter;
    const fromCard = Boolean(button.closest("[data-note-card]"));
    // Clicking the active filter in the panel toggles it off; a card chip always selects.
    const next = filter && filter === current && !fromCard ? "" : filter;
    apply(next, true);
    if (fromCard && list) {
      const top = list.getBoundingClientRect().top + window.scrollY - 140;
      if (window.scrollY > top) {
        window.scrollTo({ top, behavior: "smooth" });
      }
    }
  });

  window.addEventListener("hashchange", () => apply(fromHash(), false));
  apply(fromHash(), false);
})();
