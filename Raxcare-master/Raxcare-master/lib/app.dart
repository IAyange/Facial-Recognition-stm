import 'package:flutter/material.dart';
import 'package:RaxCare/screens/profile_screen.dart';
import 'package:RaxCare/screens/home/home.dart';
import 'package:RaxCare/screens/journal/journal.dart';
import 'package:RaxCare/screens/form/relapsehistory.dart';
import 'package:RaxCare/screens/goals/goals.dart';

class App extends StatefulWidget {
  const App({Key? key}) : super(key: key);

  @override
  _AppState createState() => _AppState();
}

class _AppState extends State<App> {
  int _currentIndex = 0;
  double _indicatorPosition = 0.0;

  // Light blue color scheme
  final Color _activeColor = Color(0xFF1976D2); // Primary blue
  final Color _inactiveColor = Color(0xFF90CAF9); // Lighter blue
  final Color _backgroundColor =
      Color(0xFFE3F2FD); // Very light blue (almost white)
  final Color _indicatorColor = Color(0xFF2196F3); // Bright blue
  final Color _navBarColor = Colors.white; // White with slight blue tint

  // Screens for navigation
  final List<Widget> _screens = [
    const HomePage(),
    RelapseHistoryScreen(),
    const GoalsScreen(),
    JournalScreen(),
    const ProfileScreen(),
  ];

  // Navigation items with custom icons
  final List<NavItem> _navItems = const [
    NavItem(icon: Icons.home_rounded, label: 'Home'),
    NavItem(icon: Icons.history_rounded, label: 'Relapses'),
    NavItem(icon: Icons.flag_rounded, label: 'Goals'),
    NavItem(icon: Icons.book_rounded, label: 'Journal'),
    NavItem(icon: Icons.person_rounded, label: 'Profile'),
  ];

  @override
  Widget build(BuildContext context) {
    final width = MediaQuery.of(context).size.width;
    final itemWidth = width / _navItems.length;

    return Scaffold(
      backgroundColor: _backgroundColor,
      body: IndexedStack(
        index: _currentIndex,
        children: _screens,
      ),
      bottomNavigationBar: Container(
        decoration: BoxDecoration(
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.1),
              blurRadius: 16,
              spreadRadius: 0,
              offset: const Offset(0, -4),
            )
          ],
        ),
        child: Stack(
          children: [
            // Background with modern rounded corners
            Container(
              height: 80 + MediaQuery.of(context).padding.bottom,
              decoration: BoxDecoration(
                color: _navBarColor,
                borderRadius: const BorderRadius.vertical(
                  top: Radius.circular(24), // More modern rounded corners
                ),
                boxShadow: [
                  BoxShadow(
                    color: Colors.blue.withOpacity(0.05),
                    blurRadius: 10,
                    spreadRadius: 2,
                  )
                ],
              ),
            ),

            // Animated indicator
            AnimatedPositioned(
              duration: const Duration(milliseconds: 300),
              curve: Curves.easeOutQuad,
              left: _indicatorPosition,
              bottom: 12 + MediaQuery.of(context).padding.bottom,
              child: Container(
                width: itemWidth * 0.6,
                height: 4,
                decoration: BoxDecoration(
                  color: _indicatorColor,
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
            ),

            // Navigation items
            Padding(
              padding: EdgeInsets.only(
                  bottom: MediaQuery.of(context).padding.bottom),
              child: Row(
                children: _navItems.asMap().entries.map((entry) {
                  final index = entry.key;
                  final item = entry.value;
                  final isActive = _currentIndex == index;

                  return Expanded(
                    child: GestureDetector(
                      onTap: () {
                        setState(() {
                          _currentIndex = index;
                          _indicatorPosition =
                              index * itemWidth + (itemWidth * 0.2);
                        });
                      },
                      behavior: HitTestBehavior.opaque,
                      child: Container(
                        height: 80,
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            // Animated icon
                            AnimatedContainer(
                              duration: const Duration(milliseconds: 300),
                              transform: Matrix4.identity()
                                ..scale(isActive ? 1.2 : 1.0),
                              child: Icon(
                                item.icon,
                                color: isActive ? _activeColor : _inactiveColor,
                                size: 28,
                              ),
                            ),
                            const SizedBox(height: 6),
                            // Animated text
                            AnimatedDefaultTextStyle(
                              duration: const Duration(milliseconds: 300),
                              style: TextStyle(
                                color: isActive ? _activeColor : _inactiveColor,
                                fontWeight: isActive
                                    ? FontWeight.w600
                                    : FontWeight.w400,
                                fontSize: isActive ? 12 : 11,
                              ),
                              child: Text(item.label),
                            ),
                          ],
                        ),
                      ),
                    ),
                  );
                }).toList(),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class NavItem {
  final IconData icon;
  final String label;

  const NavItem({
    required this.icon,
    required this.label,
  });
}
