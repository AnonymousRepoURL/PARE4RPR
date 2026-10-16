"""
greenery supports the following regex features:
  1. VariablePattern: a, b, @
  2. AlternationPattern: ab
  3. AlternationPattern: a|b
  4. GroupPattern: (ab)
  5. Non-capturing group syntax: (?:ab)
  6. Named group syntax: (?P<name>ab)
  7. Character class: [abc]
  8. Negation character class: [^abc]
  9. Range character class: [a-z]
  10.wildcard: .
  11.Shorthand character class: \\d, \\D, \\w, \\W, \\s, \\S
  12.QuantifiedPattern (without greedy/lazy): ?, *, +, {m}, {m,n}, {m,}
  13.EmptyPattern: '', (), a|, |a

greenery cannot support
  1.inert quantifiers (can parse, but quantifiers are treated as greedy quantifiers): a*?, a+?, a??, a{m,n}?, a{m,}?, a{m}?
  2.backreferences (cannot parse): \1, \2, ...
  3.assertions (cannot parse): (?=ab), (?!ab), (?<=ab), (?<!ab)
  4.anchors (cannot parse): ^, $

-----------------------------------------------------------------------
|           fsm1            |          fsm2            | equivalent? |
-----------------------------------------------------------------------
| "((x|y)?(x|y|z)+(y|z)?)+" | "((x|y)?(x|y|z)(y|z)?)+"  |   Yes       |
| "[(ab)|(ac)]"             | "[(ab)|c]"                |   Yes       |
| "a((x|y)+(x|y|z)*)+a"     | "a((x|y)(x|y|z)*)+a"      |   Yes       |
"""

from greenery import parse
from greenery.rxelems import from_fsm

# regex1 = "(a+b+)+"
# regex2 = "a(a|b)*b"
# regex1 = "(a?b{1,}c?)+"
# regex2 = "(a?bc?)+"

# if parse(regex1).to_fsm() == parse(regex2).to_fsm():
#     print("Two regular expressions are equivalent")
# else:
#     print("Caution, two regular expressions are not equivalent")


# ((a|c)+(a|b)+)+
cur_regex = "[ab]+[ac]+"  # a*b[ab]*
fsm = parse(cur_regex).to_fsm().reduce()
# Convert DFA back to greenery pattern for human-friendly simplification.
simplified = from_fsm(fsm)
print(f'simplified regex: "{simplified}"')
