#!/usr/bin/env python3
"""
Build Dota 2 RAG knowledge base.

This script fetches Dota 2 hero and item information and builds a
searchable knowledge base for the voice assistant.

Usage:
    python scripts/build_dota2_rag.py

The knowledge base will be stored persistently and used automatically
by the voice assistant when RAG is enabled.
"""

import sys
import json
import urllib.request
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from jetson_assistant.rag import RAGPipeline, WebLoader
from jetson_assistant.rag.loaders import Document


def fetch_opendota_heroes() -> list[Document]:
    """Fetch hero data from OpenDota API."""
    print("Fetching hero data from OpenDota API...")

    url = "https://api.opendota.com/api/heroes"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Dota2RAGBot/1.0)"})
    with urllib.request.urlopen(req, timeout=30) as response:
        heroes = json.loads(response.read().decode())

    documents = []
    for hero in heroes:
        # Create a document for each hero
        name = hero.get("localized_name", hero.get("name", "Unknown"))
        primary_attr = hero.get("primary_attr", "unknown")
        attack_type = hero.get("attack_type", "unknown")
        roles = ", ".join(hero.get("roles", []))

        content = f"""Hero: {name}
Primary Attribute: {primary_attr}
Attack Type: {attack_type}
Roles: {roles}

{name} is a {primary_attr} hero in Dota 2 who is typically played as {roles}. This hero uses {attack_type} attacks."""

        documents.append(Document(
            content=content,
            metadata={
                "source": "opendota_api",
                "type": "hero",
                "hero_name": name,
            }
        ))

    print(f"  Fetched {len(documents)} heroes")
    return documents


def fetch_dota_wiki_pages() -> list[Document]:
    """Fetch key pages from Dota 2 wiki."""
    print("Fetching Dota 2 wiki pages...")

    # Key wiki pages for Dota 2 knowledge
    wiki_urls = [
        "https://dota2.fandom.com/wiki/Items",
        "https://dota2.fandom.com/wiki/Mechanics",
        "https://dota2.fandom.com/wiki/Attributes",
        "https://dota2.fandom.com/wiki/Attack_damage",
        "https://dota2.fandom.com/wiki/Armor",
        "https://dota2.fandom.com/wiki/Black_King_Bar",
        "https://dota2.fandom.com/wiki/Blink_Dagger",
    ]

    documents = []
    for url in wiki_urls:
        try:
            loader = WebLoader(url, follow_links=False)
            docs = loader.load_all()
            documents.extend(docs)
            print(f"  Fetched: {url.split('/')[-1]}")
        except Exception as e:
            print(f"  Failed: {url} - {e}")

    return documents


def add_curated_knowledge() -> list[Document]:
    """Add curated Dota 2 knowledge for common questions."""
    print("Adding curated Dota 2 knowledge...")

    curated = [
        # Hero counters
        """Anti-Mage Counters:
Anti-Mage is countered by heroes with strong lockdown and burst damage. Good counters include:
- Phantom Assassin: High burst physical damage before AM can react
- Faceless Void: Chronosphere prevents Blink escape
- Lion: Hex and Mana Drain, plus Finger of Death burst
- Axe: Berserker's Call goes through spell immunity
- Bloodseeker: Rupture punishes Blink usage
Build Orchid Malevolence or Scythe of Vyse to silence/disable him.""",

        """Phantom Lancer Counters:
Phantom Lancer creates many illusions and is countered by AoE damage. Good counters include:
- Earthshaker: Echo Slam deals massive damage with many illusions
- Ember Spirit: Sleight of Fist hits all illusions
- Leshrac: Pulse Nova and Lightning Storm clear illusions
- Sven: Great Cleave destroys illusion armies
- Legion Commander: Duel targets the real hero
Build Mjollnir, Battle Fury, or Radiance for AoE damage.""",

        """Invoker Counters:
Invoker is a complex hero countered by gap closers and silences. Good counters include:
- Anti-Mage: Blink on top of him, Mana Break burns his mana
- Nyx Assassin: Mana Burn deals massive damage, Spiked Carapace reflects
- Clockwerk: Hookshot initiation, Battery Assault interrupts
- Spirit Breaker: Charge prevents channeling, Bash interrupts
Build Orchid Malevolence for silence, BKB for spell immunity.""",

        # Items
        """Black King Bar (BKB):
Black King Bar grants Spell Immunity when activated. Key facts:
- Blocks most magical damage and disables
- Duration starts at 9 seconds, decreases with each use (minimum 5 seconds)
- Does NOT block: Chronosphere, Black Hole, Duel, Roshan attacks
- Essential on most carry heroes in late game
- Costs 4050 gold (Ogre Axe + Mithril Hammer + Recipe)""",

        """Blink Dagger:
Blink Dagger allows instant teleportation up to 1200 units. Key facts:
- 15 second cooldown
- Cannot be used for 3 seconds after taking damage from heroes/Roshan
- Essential for initiators like Earthshaker, Axe, Enigma
- No mana cost
- Costs 2250 gold""",

        # Roles and positions
        """Dota 2 Positions:
Position 1 (Safe Lane Carry): Farms priority, late game damage dealer. Examples: Anti-Mage, Phantom Assassin, Faceless Void
Position 2 (Mid Lane): Solo experience, tempo controller. Examples: Invoker, Storm Spirit, Queen of Pain
Position 3 (Offlane): Initiator, frontliner. Examples: Axe, Mars, Tidehunter
Position 4 (Soft Support): Roamer, playmaker. Examples: Earth Spirit, Tusk, Mirana
Position 5 (Hard Support): Ward buyer, babysitter. Examples: Crystal Maiden, Lion, Shadow Shaman""",

        # Beginner tips
        """Best Heroes for Beginners:
Mid Lane: Viper (simple, tanky), Sniper (long range), Zeus (press R to contribute)
Carry: Wraith King (one active ability), Juggernaut (forgiving), Phantom Assassin (fun crits)
Offlane: Bristleback (tanky, simple), Axe (clear initiation), Tidehunter (big ultimate)
Support: Crystal Maiden (aura helps team), Lion (simple disables), Ogre Magi (tanky, simple)""",

        # =====================
        # NEWER HERO GUIDES
        # =====================

        # Kez (Released 2024) - Newest hero
        """How to Play Kez:
Kez is an agility carry released in 2024, the newest Dota 2 hero. He excels at chasing and bursting single targets.

Abilities:
- Talon Toss: Throws his weapon, dealing damage and slowing enemies
- Grapple: Dash to a target, dealing damage on arrival
- Shodo Sai: Passive that grants attack speed and movement speed after using abilities
- Raptor Dance (Ultimate): Unleashes rapid slashes in an area, dealing massive physical damage

How to play:
- Lane: Usually played as Position 1 carry or Position 2 mid
- Early game: Focus on last hits, use Grapple to escape ganks or secure kills
- Mid game: Farm aggressively, join fights when ultimate is ready
- Late game: Target enemy supports and squishy heroes with Grapple + Raptor Dance combo

Item build: Power Treads, Battle Fury or Maelstrom, Desolator, BKB, Butterfly, Satanic
Counters to Kez: Heroes with hard lockdown like Lion, Shadow Shaman, Bane. Ghost Scepter counters his physical damage.""",

        """Kez Counters:
Kez is countered by heroes with instant disables and tanky frontliners:
- Lion: Hex stops Kez instantly, Finger bursts him down
- Shadow Shaman: Shackles hold Kez in place during Raptor Dance
- Bane: Fiend's Grip completely shuts down Kez
- Axe: Berserker's Call interrupts Raptor Dance, Blade Mail reflects damage
- Centaur Warrunner: Tanky, Retaliate punishes Kez's fast attacks
Build Ghost Scepter or Ethereal Blade to become immune to his physical damage.""",

        # Ringmaster (Released 2024)
        """How to Play Ringmaster:
Ringmaster is a support hero released in 2024. He's a circus-themed hero with crowd control and utility.

Abilities:
- Tame the Beasts: Summons circus animals to fight for you
- Impalement Arts: Throws knives that stun enemies in a line
- Escape Act: Creates a trapdoor for allies to escape through
- The Finale (Ultimate): Big AoE disable that sets up team fights

How to play:
- Lane: Position 4 or 5 support
- Early game: Harass with Impalement Arts, protect your carry
- Mid game: Set up kills with Impalement Arts stun, use Escape Act to save allies
- Late game: Use The Finale to initiate team fights, peel for your carry

Item build: Arcane Boots, Aether Lens, Glimmer Cape, Force Staff, Aghanim's Scepter
Ringmaster pairs well with: Heroes who can follow up on his stuns like Sven, Phantom Assassin, Juggernaut""",

        # Muerta (Released 2023)
        """How to Play Muerta:
Muerta is an intelligence carry released in 2023. She's a ghostly gunslinger with high magic damage.

Abilities:
- Dead Shot: Fires a bullet that fears enemies and deals damage
- The Calling: Summons spirits that silence and damage enemies in an area
- Gunslinger: Passive that gives bonus attack damage and double-shot chance
- Pierce the Veil (Ultimate): Become ethereal, your attacks deal magic damage, immune to physical

How to play:
- Lane: Position 1 carry or Position 2 mid
- Early game: Use Dead Shot to harass and secure last hits, fear is strong in lane
- Mid game: Farm with Gunslinger procs, join fights with ultimate for magic burst
- Late game: Pierce the Veil makes you immune to physical carries, focus their supports

Item build: Power Treads, Gleipnir, Aghanim's Scepter, BKB, Revenant's Brooch, Bloodthorn
Counters to Muerta: Magic damage dealers (she's vulnerable to magic during ult), Silencer, Anti-Mage""",

        """Muerta Counters:
Muerta is countered by magic damage and silences:
- Silencer: Global Silence stops her ultimate and abilities
- Anti-Mage: Mana Break drains her mana, Counterspell reflects Dead Shot
- Pugna: Decrepify makes allies immune to her ultimate damage
- Nyx Assassin: Mana Burn deals huge damage to her high mana pool
- Skywrath Mage: Pure magic burst destroys her during Pierce the Veil
Build Orchid/Bloodthorn to silence her before she can ult.""",

        # Primal Beast (Released 2022)
        """How to Play Primal Beast:
Primal Beast is a strength offlaner released in 2022. He's a massive beast that excels at initiation and disruption.

Abilities:
- Onslaught: Charge forward, knocking back and damaging enemies
- Trample: Stomp the ground repeatedly, dealing AoE damage around you
- Uproar: Passive that stacks damage and armor when taking damage
- Pulverize (Ultimate): Grab an enemy and slam them into the ground repeatedly

How to play:
- Lane: Position 3 offlane
- Early game: Trade hits to build Uproar stacks, use Onslaught to escape or chase
- Mid game: Initiate with Onslaught into Pulverize on key targets
- Late game: Be the frontliner, grab enemy carry with Pulverize, soak damage for Uproar

Item build: Phase Boots, Vanguard into Crimson Guard, BKB, Aghanim's Scepter, Heart of Tarrasque
Primal Beast combos: Onslaught from fog -> Pulverize -> Trample while team follows up""",

        """Primal Beast Counters:
Primal Beast is countered by kiting and disables that interrupt his abilities:
- Venomancer: Slows prevent Primal Beast from reaching targets
- Viper: Nethertoxin breaks his Uproar passive
- Shadow Demon: Disruption interrupts Pulverize, Demonic Purge slows him
- Lifestealer: Rage makes him immune to Pulverize, Feast heals through Trample
- Ursa: Can man-fight Primal Beast and burst him before Uproar stacks
Avoid grouping up against his Trample, spread out in fights.""",

        # Marci (Released 2021)
        """How to Play Marci:
Marci is a strength hero released in 2021. She's a versatile fighter who can be played as carry or support.

Abilities:
- Dispose: Grab an ally or enemy and throw them, stunning enemies hit
- Rebound: Dash to an ally and jump to a target location, stunning enemies
- Sidekick: Buff an ally with lifesteal and bonus damage
- Unleash (Ultimate): Enter a flurry state with rapid attacks and bonus movement speed

How to play as Carry (Pos 1/2):
- Farm aggressively, use Rebound to escape ganks
- In fights, use Rebound to initiate, Unleash for sustained damage
- Item build: Phase Boots, Desolator, BKB, Skull Basher into Abyssal, Satanic

How to play as Support (Pos 4):
- Use Dispose to save allies or throw enemies into your team
- Sidekick your carry for extra damage and sustain
- Item build: Tranquil Boots, Medallion, Force Staff, Aghanim's Scepter""",

        # Dawnbreaker (Released 2021)
        """How to Play Dawnbreaker:
Dawnbreaker is a strength hero released in 2021. She's a global presence offlaner with healing and damage.

Abilities:
- Starbreaker: Spin your hammer, dealing damage and stunning on the final hit
- Celestial Hammer: Throw your hammer, then pull yourself to it dealing damage
- Luminosity: Passive that heals nearby allies when you land critical hits
- Solar Guardian (Ultimate): Channel then land at a target location globally, healing allies and damaging enemies

How to play:
- Lane: Position 3 offlane
- Early game: Trade with Starbreaker, heal yourself and allies with Luminosity
- Mid game: Look for Solar Guardian opportunities across the map
- Late game: Frontline in fights, use Solar Guardian to turn fights anywhere on the map

Item build: Phase Boots, Echo Sabre, BKB, Desolator, Assault Cuirass, Satanic
Key tip: Always watch the minimap for Solar Guardian opportunities to save allies or join fights""",

        # More counters for popular heroes
        """Pudge Counters:
Pudge relies on landing Meat Hook. Counter him with:
- Lifestealer: Rage makes you immune to Dismember
- Juggernaut: Blade Fury makes you immune to Hook and Dismember
- Anti-Mage: Blink away after getting hooked
- Slark: Pounce escapes, Dark Pact purges Dismember
Buy Linken's Sphere to block Hook or Force Staff to save hooked allies.""",

        """Tinker Counters:
Tinker is countered by heroes who can catch him in trees:
- Spirit Breaker: Charge finds him anywhere, Bash cancels Rearm
- Storm Spirit: Ball Lightning closes distance instantly
- Spectre: Haunt finds Tinker anywhere, Reality to jump on him
- Zeus: Thundergod's Wrath reveals him, Nimbus provides vision
- Clockwerk: Hookshot catches him in trees, Cogs trap him
Buy Blade Mail to reflect Laser and Rocket damage.""",

        """Sniper Counters:
Sniper has no escape and low HP. Counter him with gap closers:
- Spirit Breaker: Charge + Nether Strike deletes him
- Phantom Assassin: Blink Strike + Coup de Grace one-shots him
- Storm Spirit: Ball Lightning to close distance
- Clockwerk: Hookshot initiation from long range
- Spectre: Haunt + Reality guarantees you reach him
Buy Blink Dagger to initiate on him, Blade Mail to reflect Assassinate.""",

        # Newest heroes release order
        """Dota 2 Newest Heroes (Release Order):
2024: Kez (newest), Ringmaster
2023: Muerta
2022: Primal Beast
2021: Marci, Dawnbreaker
2020: Hoodwink, Dawnbreaker
2019: Snapfire, Void Spirit
2018: Grimstroke, Mars

Kez is the newest hero in Dota 2, released in late 2024. He is an agility carry with high mobility and burst physical damage.""",

        # Popular item explanations
        """Aghanim's Scepter:
Aghanim's Scepter upgrades a hero's abilities, usually their ultimate. Key facts:
- Costs 4200 gold (Point Booster + Ogre Axe + Blade of Alacrity + Staff of Wizardry)
- Effect is different for every hero - check each hero's ability description
- Can be consumed with Aghanim's Blessing to free up the slot
- Some heroes get new abilities entirely (e.g., Invoker gets two extra Invoke charges)""",

        """Divine Rapier:
Divine Rapier is the highest damage item in Dota 2. Key facts:
- Costs 5950 gold, gives +350 damage
- DROPS ON DEATH and can be picked up by enemies
- Once dropped, it becomes "free" and doesn't drop again if the new owner dies
- Only buy when desperate or extremely far ahead
- Build on heroes with good survivability: Medusa, Gyrocopter, Ember Spirit""",

        """Refresher Orb:
Refresher Orb resets all ability and item cooldowns. Key facts:
- Costs 5000 gold
- 180 second cooldown
- Costs 350 mana to activate
- Essential on heroes with long-cooldown ultimates: Tidehunter, Enigma, Magnus
- Allows double BKB, double ultimate, etc.""",
    ]

    documents = []
    for text in curated:
        documents.append(Document(
            content=text,
            metadata={
                "source": "curated",
                "type": "knowledge",
            }
        ))

    print(f"  Added {len(documents)} curated entries")
    return documents


def main():
    print("=" * 60)
    print("Building Dota 2 RAG Knowledge Base")
    print("=" * 60)
    print()

    # Create RAG pipeline
    rag = RAGPipeline(
        collection_name="dota2",
        embedding_model="minilm",
        chunk_strategy="paragraph",
        chunk_size=500,
    )

    # Check if already built
    existing_count = rag.count()
    if existing_count > 0:
        print(f"Existing knowledge base found with {existing_count} chunks.")
        response = input("Rebuild from scratch? (y/N): ").strip().lower()
        if response == "y":
            print("Clearing existing data...")
            rag.clear()
        else:
            print("Keeping existing data. Use --clear flag to force rebuild.")
            print("\nCurrent sources:")
            for source in rag.get_sources():
                print(f"  - {source}")
            return

    print()

    # Fetch and ingest data
    all_documents = []

    # 1. OpenDota API data
    try:
        all_documents.extend(fetch_opendota_heroes())
    except Exception as e:
        print(f"  Failed to fetch OpenDota data: {e}")

    # 2. Curated knowledge
    all_documents.extend(add_curated_knowledge())

    # 3. Wiki pages (optional, slower)
    try:
        all_documents.extend(fetch_dota_wiki_pages())
    except Exception as e:
        print(f"  Failed to fetch wiki pages: {e}")

    print()
    print(f"Total documents collected: {len(all_documents)}")

    # Chunk and add to vector store
    print("\nProcessing and indexing...")
    chunks = rag.chunker.chunk_documents(all_documents)
    print(f"Created {len(chunks)} chunks")

    count = rag.store.add(chunks)
    print(f"Added {count} chunks to vector store")

    # Summary
    print()
    print("=" * 60)
    print("Knowledge Base Ready!")
    print("=" * 60)
    print(f"Collection: {rag.collection_name}")
    print(f"Total chunks: {rag.count()}")
    print(f"Sources: {len(rag.get_sources())}")
    print()
    print("Test query:")
    results = rag.retrieve("Who counters Anti-Mage?", top_k=2)
    for r in results:
        print(f"  [{r['score']:.2f}] {r['content'][:100]}...")
    print()
    print("The voice assistant will now use this knowledge base!")
    print("Restart the assistant to enable RAG.")


if __name__ == "__main__":
    main()
