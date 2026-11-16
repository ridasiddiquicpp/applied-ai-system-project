import hashlib

import streamlit as st
from pawpal_system import Owner, Pet, Scheduler, Task
from ai_parser import (
    AIParserError, Clarification, backend_label, parse_and_validate,
)
from ai_parser import next_pet_id as _next_pet_id, next_task_id as _next_task_id
from transcribe import TranscribeError, transcribe
from datetime import date, datetime

st.set_page_config(page_title="PawPal+", page_icon="🐾", layout="centered")

if "owner" not in st.session_state:
    st.session_state.owner = Owner(1, "")

owner = st.session_state.owner

st.title("🐾 PawPal+")

st.markdown(
    """
Welcome to the PawPal+ starter app.

This file is intentionally thin. It gives you a working Streamlit app so you can start quickly,
but **it does not implement the project logic**. Your job is to design the system and build it.

Use this app as your interactive demo once your backend classes/functions exist.
"""
)

with st.expander("Scenario", expanded=True):
    st.markdown(
        """
**PawPal+** is a pet care planning assistant. It helps a pet owner plan care tasks
for their pet(s) based on constraints like time, priority, and preferences.

You will design and implement the scheduling logic and connect it to this Streamlit UI.
"""
    )

with st.expander("What you need to build", expanded=True):
    st.markdown(
        """
At minimum, your system should:
- Represent pet care tasks (what needs to happen, how long it takes, priority)
- Represent the pet and the owner (basic info and preferences)
- Build a plan/schedule for a day that chooses and orders tasks based on constraints
- Explain the plan (why each task was chosen and when it happens)
"""
    )

st.divider()


def next_task_id() -> int:
    """An id no existing task uses (count + 1 can collide once recurring tasks are added)."""
    return _next_task_id(owner)


def next_pet_id() -> int:
    """An id no existing pet uses."""
    return _next_pet_id(owner)


def show_table(rows: list) -> None:
    """Show a list of dicts as a markdown table.

    Used instead of st.table, which needs pandas and pyarrow; Windows
    Application Control can block pyarrow's DLLs and crash the page.
    """
    if not rows:
        return
    headers = list(rows[0])

    def cell(value) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines += ["| " + " | ".join(cell(row[h]) for h in headers) + " |" for row in rows]
    st.markdown("\n".join(lines))


# ---------------------------------------------------------------- Quick add
# Say or type a request. Nothing changes until the user clicks Confirm.
st.subheader("🎙️ Quick add")
st.caption(
    "Record or type a request, like \"walk Mochi every day at 8am for 30 minutes\", "
    "\"add a cat named Luna\" or \"remove Max\". Check the text, click Parse, "
    "then confirm the preview."
)
st.caption(f"AI model in use: {backend_label()}")

for kind, message in st.session_state.pop("quick_msgs", []):
    getattr(st, kind)(message)

if st.session_state.pop("quick_clear", False):
    st.session_state.quick_text = ""  # must happen before the text box is created
nonce = st.session_state.setdefault("quick_nonce", 0)

audio = st.audio_input("Record a request", key=f"quick_audio_{nonce}")
if audio is not None:
    audio_bytes = audio.getvalue()
    audio_id = hashlib.md5(audio_bytes).hexdigest()
    if st.session_state.get("quick_audio_seen") != audio_id:  # reruns must not re-transcribe
        st.session_state.quick_audio_seen = audio_id
        try:
            with st.spinner("Transcribing (the first time downloads the speech model)..."):
                st.session_state.quick_text = transcribe(audio_bytes, [p.name for p in owner.pets])
        except TranscribeError as error:
            st.error(str(error))

quick_text = st.text_area("Or type a request (you can edit the transcript here)", key="quick_text")

if st.button("Parse"):
    st.session_state.quick_pending = None
    st.session_state.quick_question = None
    st.session_state.quick_corrections = []
    try:
        with st.spinner("Understanding your request..."):
            # The model proposes; the code checks it against your pets and, if the
            # check fails, sends the error back so the model can fix its own answer.
            agent = parse_and_validate(quick_text, owner, date.today())
        st.session_state.quick_corrections = agent.corrections
        if isinstance(agent.outcome, Clarification):
            st.session_state.quick_question = agent.outcome.question
        else:
            st.session_state.quick_pending = agent.outcome
    except AIParserError as error:
        st.error(str(error))

if st.session_state.get("quick_question"):
    st.info(f"{st.session_state.quick_question}  (Add your answer to the text above and click Parse again.)")

for note in st.session_state.get("quick_corrections", []):
    st.caption(f"🔁 The AI corrected itself after: {note}")

pending = st.session_state.get("quick_pending")
if pending:
    action = pending[0]
    st.markdown("**Here's what I understood:**")
    if action == "task":
        _, pet, task = pending
        st.info(
            f"Add task for **{pet.name}**: {task.description}, {task.frequency}, "
            f"{task.due_time:%b %d, %I:%M %p}, {task.duration_minutes} min, {task.priority} priority"
        )
    elif action == "add_pet":
        new_pet = pending[1]
        details = [new_pet.species] + ([new_pet.breed] if new_pet.breed else []) + (
            [f"age {new_pet.age}"] if new_pet.age else []
        )
        st.info(f"Add pet **{new_pet.name}** ({', '.join(details)})")
    else:
        _, pet, task_count = pending
        st.warning(f"Remove **{pet.name}**? This also removes its {task_count} task(s) and cannot be undone.")

    confirm_col, cancel_col = st.columns(2)
    confirmed = confirm_col.button("Confirm", type="primary")
    cancelled = cancel_col.button("Cancel")
    if confirmed or cancelled:
        messages = []
        if confirmed:
            if action == "task":
                try:
                    warnings = Scheduler(owner).add_task(pet, task)
                    messages.append(("success", f"Added {task.description} for {pet.name}"))
                    messages += [("warning", w) for w in warnings]
                except ValueError as error:
                    messages.append(("error", f"Task not added: {error}"))
            elif action == "add_pet":
                owner.add_pet(new_pet)
                messages.append(("success", f"Added pet {new_pet.name}"))
            else:
                owner.remove_pet(pet)
                messages.append(("success", f"Removed {pet.name} and its {task_count} task(s)"))
        else:
            messages.append(("info", "Cancelled. Nothing was changed."))
        st.session_state.quick_msgs = messages
        st.session_state.quick_pending = None
        st.session_state.quick_question = None
        st.session_state.quick_clear = True
        st.session_state.quick_nonce = nonce + 1  # new key resets the recorder
        st.rerun()

st.divider()


st.subheader("Basic Information")
owner_name = st.text_input("Owner name", value="Jordan")
owner.name = owner_name
# st.write(owner)


pet_name = st.text_input("Pet name", value="Mochi")
species = st.selectbox("Species", ["dog", "cat", "other"])
if st.button("Add pet"):
    pet = Pet(next_pet_id(), pet_name, species)
    owner.add_pet(pet)

st.markdown("### Tasks")
st.caption("Add a few tasks. In your final version, these should feed into your scheduler.")

if "tasks" not in st.session_state:
    st.session_state.tasks = []

if not owner.pets:
    st.warning("Add a pet first, then you can add tasks.")

else:
    pet_index = st.selectbox(
        "Pet", range(len(owner.pets)), format_func=lambda i: owner.pets[i].name
    )
    pet_choice = owner.pets[pet_index]



    col1, col2, col3 = st.columns(3)
    with col1:
        task_title = st.text_input("Task title", value="Morning walk")
    with col2:
        duration = st.number_input("Duration (minutes)", min_value=1, max_value=240, value=20)
    with col3:
        priority = st.selectbox("Priority", ["low", "medium", "high"], index=2)

    col4, col5, col6 = st.columns(3)

    with col4:
        task_date = st.date_input("Date")
    with col5:
        task_time = st.time_input("Time")
    with col6:
        frequency = st.selectbox("Frequency", ["once", "daily", "weekly"])


    # Inside the else: pet_choice and the form fields only exist when there is a pet.
    if st.button("Add task"):
        due = datetime.combine(task_date, task_time)
        task = Task(
            next_task_id(),
            task_title,
            "general",
            due,
            int(duration),
            priority,
            frequency,
        )
        try:
            warnings = Scheduler(owner).add_task(pet_choice, task)
        except ValueError as error:
            st.error(f"Task not added: {error}")
        else:
            st.success(f"Added {task.description} for {pet_choice.name}")
            for warning in warnings:
                st.warning(warning)


scheduler = Scheduler(owner)

warnings = scheduler.get_conflict_warnings()
if warnings:
    st.markdown("### ⚠️ Scheduling Conflicts")
    for message in warnings:
        st.warning(message)
elif owner.get_all_tasks():
    st.success("No scheduling conflicts. Your schedule looks good!")


st.divider()

st.markdown("### Your Tasks")
if "complete_msg" in st.session_state:
    st.success(st.session_state.pop("complete_msg"))

if owner.get_all_tasks():
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        sort_by = st.radio("Sort by", ["Time", "Priority"], horizontal=True)
    with col_b:
        pet_filter = st.selectbox("Pet filter", ["All"] + [p.name for p in owner.pets])
    with col_c:
        status_filter = st.selectbox("Status", ["All", "Incomplete", "Complete"])

    
    incomplete = scheduler.filter_tasks(is_complete=False)
    if incomplete:
        st.markdown("**Mark a task complete**")
        select_col, button_col = st.columns([4, 1])
        with select_col:
            task_index = st.selectbox(
                "Task to complete",
                range(len(incomplete)),
                format_func=lambda i: f"{incomplete[i].pet.name} - {incomplete[i].description} ({incomplete[i].due_time:%b %d, %I:%M %p})",
                label_visibility="collapsed",
            )
        with button_col:
            mark_clicked = st.button("Mark complete")

        if mark_clicked:
            next_task = scheduler.mark_task_complete(incomplete[task_index])
            message = f"Marked '{incomplete[task_index].description}' complete."
            if next_task:
                message += f" Next {next_task.frequency} task scheduled for {next_task.due_time:%b %d, %I:%M %p}."
            st.session_state.complete_msg = message
            st.rerun()

    if sort_by == "Priority":
        sorted_tasks = scheduler.sort_by_priority()
    else:
        sorted_tasks = scheduler.sort_by_time()

    wanted = scheduler.filter_tasks(
        is_complete={"All": None, "Incomplete": False, "Complete": True}[status_filter],
        pet_name=None if pet_filter == "All" else pet_filter,
    )
    sorted_tasks = [t for t in sorted_tasks if any(t is w for w in wanted)]

    if sorted_tasks:
        show_table(
            [
                {
                    "Date": f"{t.due_time:%b %d}",
                    "Time": f"{t.due_time:%I:%M %p}",
                    "Pet": t.pet.name,
                    "Task": t.description,
                    "Priority": str(t.priority),
                    "Duration (min)": t.duration_minutes,
                    "Done": "✅" if t.is_complete else "",
                }
                for t in sorted_tasks
            ]
        )
    else:
       st.info("No tasks match these filters.") 
else:
    st.info("No tasks yet. Add one above.")



st.subheader("Build Schedule")
st.caption("Pick a day to see its tasks in time order.")

schedule_day = st.date_input("Schedule for", key="schedule_day")

if st.button("Generate schedule"):
    todays_tasks = scheduler.get_daily_schedule(schedule_day)
    if todays_tasks:
        st.success(f"{len(todays_tasks)} task(s) scheduled for {schedule_day:%B %d, %Y}")
        show_table(
            [
                {
                    "Time": f"{t.due_time:%I:%M %p}",
                    "Pet": t.pet.name,
                    "Task": t.description,
                    "Priority": str(t.priority),
                    "Duration (min)": t.duration_minutes,
                    "Done": "✅" if t.is_complete else "",

                }
                for t in todays_tasks
            ]
        )
    else:
        st.info(f"Nothing scheduled for {schedule_day:%B %d, %Y}.")

