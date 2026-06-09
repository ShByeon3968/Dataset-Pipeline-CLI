"""
Class CRUD service

Color assignment guarantees no two classes in the same dataset share the same color.
Strategy:
  1. Query all colors already in use for the dataset.
  2. Pick the first color from CLASS_COLORS not yet taken.
  3. If all predefined colors are exhausted, generate a unique color using the
     golden-angle hue distribution (evenly spaced across the hue wheel).
"""
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models import Class

# 30 visually distinct colors — enough for typical dataset class counts
CLASS_COLORS = [
    "#E53935",  # red
    "#1E88E5",  # blue
    "#43A047",  # green
    "#FB8C00",  # orange
    "#8E24AA",  # purple
    "#00ACC1",  # cyan
    "#F4C430",  # yellow
    "#EC407A",  # pink
    "#00897B",  # teal
    "#6D4C41",  # brown
    "#3949AB",  # indigo
    "#7CB342",  # light green
    "#F06292",  # light pink
    "#26A69A",  # medium teal
    "#AB47BC",  # medium purple
    "#FF7043",  # deep orange
    "#5C6BC0",  # medium indigo
    "#D4E157",  # lime
    "#26C6DA",  # light cyan
    "#FFA726",  # light orange
    "#EF5350",  # light red
    "#42A5F5",  # light blue
    "#66BB6A",  # light green 2
    "#FFCA28",  # amber
    "#7E57C2",  # medium purple 2
    "#0288D1",  # light blue 2
    "#FF5722",  # deep orange 2
    "#78909C",  # blue grey
    "#8D6E63",  # medium brown
    "#9CCC65",  # light lime
]


def _generate_color(index: int) -> str:
    """
    Generate a unique color for overflow cases (> 30 classes).
    Uses the golden-angle step on the hue wheel for even perceptual spacing.
    """
    hue = (index * 137.508) % 360   # golden angle ≈ 137.5°
    saturation = 65 + (index % 3) * 5   # 65 / 70 / 75 %
    lightness = 48 + (index % 2) * 6    # 48 / 54 %
    return f"hsl({hue:.0f},{saturation}%,{lightness}%)"


async def get_or_create_class(db: AsyncSession, dataset_id: int, name: str) -> Class:
    # Return existing class if already present
    stmt = select(Class).where(Class.dataset_id == dataset_id, Class.name == name)
    result = await db.execute(stmt)
    cls = result.scalar_one_or_none()
    if cls:
        return cls

    # Collect all colors already in use for this dataset
    used_res = await db.execute(
        select(Class.color).where(Class.dataset_id == dataset_id)
    )
    used_colors: set[str] = {row[0] for row in used_res.all()}

    # Pick first predefined color not yet taken
    color: str | None = None
    for c in CLASS_COLORS:
        if c not in used_colors:
            color = c
            break

    # Overflow: generate a unique hue-shifted color
    if color is None:
        idx = len(used_colors)
        while True:
            candidate = _generate_color(idx)
            if candidate not in used_colors:
                color = candidate
                break
            idx += 1

    cls = Class(dataset_id=dataset_id, name=name, color=color)
    db.add(cls)
    await db.flush()
    return cls
