import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../chat.dart';
import '../../journal/journal.dart';
import '../../sponsor/sponsor.dart';
import '../../sponsor/support.dart';

class QuickActions extends StatelessWidget {
  const QuickActions({Key? key}) : super(key: key);

  Widget _buildQuickActionButton({
    required IconData icon,
    required String label,
    required Color color,
    required VoidCallback onPressed,
  }) {
    return GestureDetector(
      onTap: onPressed,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: color.withOpacity(0.15),
              borderRadius: BorderRadius.circular(20),
              boxShadow: [
                BoxShadow(
                  color: color.withOpacity(0.2),
                  blurRadius: 10,
                  offset: const Offset(0, 4),
                ),
              ],
              gradient: LinearGradient(
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
                colors: [
                  color.withOpacity(0.2),
                  color.withOpacity(0.1),
                ],
              ),
            ),
            child: Icon(
              icon,
              color: color,
              size: 30,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            label,
            textAlign: TextAlign.center,
            style: GoogleFonts.poppins(
              fontSize: 14,
              fontWeight: FontWeight.w500,
              color: Colors.grey[800],
              height: 1.2,
            ),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 8.0),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceAround,
        children: [
          _buildQuickActionButton(
            icon: Icons.phone_in_talk,
            label: 'Call\nSponsor',
            color: Colors.green[700]!,
            onPressed: () {
              Navigator.of(context).push(
                MaterialPageRoute(
                    builder: (context) => const CallSponsorScreen()),
              );
            },
          ),
          _buildQuickActionButton(
            icon: Icons.edit_note,
            label: 'Daily\nJournal',
            color: Colors.purple[700]!,
            onPressed: () {
              Navigator.of(context).push(
                MaterialPageRoute(builder: (context) => JournalScreen()),
              );
            },
          ),
          _buildQuickActionButton(
            icon: Icons.people_alt,
            label: 'Support\nGroup',
            color: Colors.orange[700]!,
            onPressed: () {
              Navigator.of(context).push(
                MaterialPageRoute(
                    builder: (context) => const SupportGroupScreen()),
              );
            },
          ),
          _buildQuickActionButton(
            icon: Icons.chat_bubble_outline,
            label: 'Chat\nSupport',
            color: Colors.blue[700]!,
            onPressed: () {
              Navigator.of(context).push(
                MaterialPageRoute(builder: (context) => ChatScreen()),
              );
            },
          ),
        ],
      ),
    );
  }
}
